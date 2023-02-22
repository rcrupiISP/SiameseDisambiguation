from numpy import (
    argsort,
    max,
    float32 as np_float32, 
    int64 as np_int64,
    union1d
)
import numpy as np
from pydantic import validate_arguments
from .alearner_base import SimulatedActiveLearner


class RandomActiveLearner(SimulatedActiveLearner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    

class BasicActiveLearner(SimulatedActiveLearner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def choose(self, preds, sigma=1/6):
        """
        Method used by the least-confident learners to choose the samples
        :param preds: predictions of the classifier over the unlabeled
        :param sigma: noise over the uncertain
        :return: index of the samples chosen as new samples to label
        """
        if self.batch_list is None:
            batch = self.batch_size
        else:
            batch = self.batch_list[self.iter_batch_list]
            self.iter_batch_list += 1

        if len(preds.shape) == 1:
            max_preds = preds
        else:
            max_preds = max(preds, axis=1)
        np.random.seed(10)
        max_preds = max_preds + np.random.randn(len(max_preds)) * sigma
        idx_chosen = argsort(max_preds)[:batch]
        return idx_chosen


class BoostActiveLearner(SimulatedActiveLearner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _one_step_predict(self):
        # choose the instances for human responses
        preds_unl = self.model.predict_proba(x=self.x_unl)
        return self.choose(preds=preds_unl, y_true=self.y_unl)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def choose(self, preds, y_true):
        if self.batch_list is None:
            batch = self.batch_size
        else:
            batch = self.batch_list[self.iter_batch_list]
            self.iter_batch_list += 1
        ae = -abs(preds[:, 1] - y_true)
        idx_chosen = argsort(ae)[:batch]#[:self.batch_size]
        return idx_chosen
