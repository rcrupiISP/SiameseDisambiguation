from numpy import (
    unique, 
    vstack, 
    hstack, 
    argmax, 
    ones, 
    float32 as np_float32, 
    int64 as np_int64,
    array
)
from numpy.random import seed as np_seed, choice
from typing import Any
from pandas import DataFrame
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    balanced_accuracy_score,
    f1_score,
)
from pydantic import validate_arguments
from tqdm import tqdm


class ModelNotValidError(Exception):
    def __init__(self, message="The model is not valid."):
        super().__init__(message)



class WrappedModel:

    def __init__(self):
        pass

    def predict_proba(self, x):
        pass

    def fit_full(
        self, 
        x,
        y
    ) -> None:
        pass

    def fit_batch(
        self, 
        x_batch,
        y_batch
    ) -> None:
        pass



class BaseALearner:

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self, 
        x_train,
        y_train,
        x_unl,
        batch_size: int, 
        seed: int, 
        model: Any,
        batch_list: Any
    ) -> None:
        self.x_train_orig_shape = x_train.shape
        self.y_train_orig_shape = y_train.shape
        self.x_train = x_train
        self.y_train = y_train
        self.x_unl = x_unl
        self.n_classes = len(unique(self.y_train))
        self.batch_size = batch_size
        self.seed = seed
        self.model = model
        self.batch_list = batch_list
        self.iter_batch_list = 0
        self.x_batch = None
        self.y_batch = None
        self.array_performances = []
        assert callable(getattr(self.model, "predict_proba", None)), \
            "Model not valid. Method .predict_proba not defined."
        assert callable(getattr(self.model, "full_fit", None)), \
            "Model not valid. Method .full_fit not defined."
        assert callable(getattr(self.model, "batch_fit", None)), \
            "Model not valid. Method .batch_fit not defined."

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def choose(
        self, 
        preds
    ):
        """
        Method used to choose the new sample to label
        :param preds: predictions over the unlabeled
        :return: index of the chosen samples
        """
        np_seed(self.seed)
        if self.batch_list is None:
            batch = self.batch_size
        else:
            batch = self.batch_list[self.iter_batch_list]
            self.iter_batch_list += 1
        idx_chosen = choice(preds.shape[0], batch, replace=False)
        return idx_chosen

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get_human_response(
        self, 
        x,
        idx,
    ):
        pass

    def _one_step_predict(self):
        # choose the instances for human responses
        if 'predict_uncertainty' in dir(self.model):
            preds_unl = self.model.predict_uncertainty(x=self.x_unl)
        else:
            preds_unl = self.model.predict_proba(x=self.x_unl)
        return self.choose(preds=preds_unl)

    @validate_arguments
    def one_step(self, fit_mode: str = "batch"):
        """
        Compute the model performance over the new batch sample (pre batch), then add the new batch samples to the
        already labelled training set and fit of the model according to fit_mode
        :param fit_mode: if batch, update the model by re-fitting on the new batch samples, else does a new fit over the
        whole training + new batch samples
        :return: update the model fitted, remove the new batch samples from the unlabeled samples
        """
        idx_chosen = self._one_step_predict()
        self._tmp_idx_chosen = idx_chosen
        x_chosen = self.x_unl[idx_chosen, :]
        y_chosen = self.get_human_response(x=x_chosen, idx=idx_chosen)
        self.x_batch = x_chosen.copy()
        self.y_batch = y_chosen.copy()
        self.array_performances.append(
            self.get_performance(x=self.x_batch, y=self.y_batch, type_set='pre_batch')
        )

        # fit
        if fit_mode == "batch":
            self.model.batch_fit(x_batch=x_chosen, y_batch=y_chosen)
            # update training set
            self.x_train = vstack((self.x_train, x_chosen))
            self.y_train = hstack((self.y_train, y_chosen))
        else:
            # update training set
            self.x_train = vstack((self.x_train, x_chosen))
            self.y_train = hstack((self.y_train, y_chosen))
            self.model.full_fit(x=self.x_train, y=self.y_train)
        # update unlabeled set
        mask = ones(self.x_unl.shape[0], dtype=bool)
        mask[idx_chosen] = False
        self._tmp_mask_unchosen = mask
        self.x_unl = self.x_unl[mask]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get_performance(
        self, 
        x,
        y,
        type_set
    ):
        """
        compute the performance over a given dataset
        :param x: data
        :param y: label
        :param type_set: type of dataset on which the performances are computed
        :return: metric performances per type_set
        """
        preds = argmax(self.model.predict_proba(x=x), axis=1)
        dict_p = {"n_lab": self.x_train.shape[0]*1.0}
        dict_p['type_set'] = type_set
        dict_p["accuracy_unweighted"] = accuracy_score(y, preds)
        dict_p["precision"] = precision_score(
            y, preds, average='macro', zero_division=0
        )
        dict_p["recall"] = recall_score(y, preds, average='macro')
        dict_p["balanced_accuracy"] = balanced_accuracy_score(y, preds)
        dict_p["f1_macro"] = f1_score(y, preds, average='macro')
        dict_p["f1"] = f1_score(y, preds)
        return dict_p


class SimulatedActiveLearner(BaseALearner):

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self, 
        x_train,
        y_train,
        x_unl,
        y_unl,
        x_test,
        y_test,
        batch_size: int, 
        seed: int, 
        model: Any,
        batch_list: Any
    ) -> None:
        super().__init__(x_train, y_train, x_unl, batch_size, seed, model, batch_list)
        self.y_unl = y_unl
        self.x_test = x_test
        self.y_test = y_test

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def get_human_response(
        self, 
        x,
        idx,
    ):
        return self.y_unl[idx]

    @validate_arguments
    def one_step(self, fit_mode: str = "batch"):
        super().one_step(fit_mode=fit_mode)
        self.y_unl = self.y_unl[self._tmp_mask_unchosen]

    @validate_arguments
    def train(self, fit_mode="full"):
        """
        Function implementing the training for the active learners. Iterates over the batch size until all the
        unlabelled samples are used.
        :param fit_mode: mode used to do the fit
        :return: a dataset with the performances for each set_type and at each iteration
        """
        self.model.full_fit(x=self.x_train, y=self.y_train)
        self.array_performances.append(
            self.get_performance(x=self.x_test, y=self.y_test, type_set='test')
        )
        self.array_performances.append(
            self.get_performance(x=self.x_unl, y=self.y_unl, type_set='unl')
        )
        if self.batch_list == None:
            number_of_it = int(self.x_unl.shape[0]/self.batch_size)
        else:
            number_of_it = len(self.batch_list)
        with tqdm(range(number_of_it)) as pbar:
            for iter in pbar:
                self.one_step(fit_mode=fit_mode)
                dict_p = self.get_performance(x=self.x_test, y=self.y_test, type_set='test')
                self.array_performances.append(
                    dict_p
                )
                if len(self.x_unl) > 0:
                    self.array_performances.append(
                        self.get_performance(x=self.x_unl, y=self.y_unl, type_set='unl')
                    )
                self.array_performances.append(
                    self.get_performance(x=self.x_batch, y=self.y_batch, type_set='post_batch')
                )
                pbar.set_description(
                    "iter = " + str(iter+1) + \
                    " | acc = " + str(round(dict_p["balanced_accuracy"], 2))
                )
        return DataFrame(self.array_performances)
