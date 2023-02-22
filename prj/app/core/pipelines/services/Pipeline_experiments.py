import pandas as pd
import os
from seaborn import lineplot, set_theme
import matplotlib.pyplot as plt
from numpy.random import seed
from prj.app.config.config import *
from prj.app.core.data.services.load_dataset_experiments import get_dataset, get_train_test
from prj.app.core.models.services.models_utils import benchmark_model_generator, siamese_model_generator, \
    models_performance
from prj.app.core.models.services.match_method import Benchmark
from prj.app.core.models.services.LSTM_class import Siamese
from prj.app.core.models.services.alearners import BasicActiveLearner, RandomActiveLearner  #, BoostActiveLearner
from prj.app.core.models.services.alearner_base import WrappedModel
seed(16)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# default source is: 'https://raw.githubusercontent.com/krishnakt031990/Crawl-Wiki-For-Acronyms/master/AcronymsFile.csv'
ground_dtf = get_dataset(PATH_EXP + 'data/',
                         frac=DATA_OPT['frac'],
                         n_neg=DATA_OPT['n_neg'],
                         source=DATA_OPT['source'],
                         df_col=X_COLS + [Y_COL])

ENCODER_OPT['encode_on'] = ground_dtf[X_COLS]

tot_needed = max(n_train.values()) + max(n_test.values())[0]
remaining = ground_dtf.shape[0] - tot_needed
assert tot_needed <= ground_dtf.shape[0], 'Invalid train and test size: not enough samples'
if max(n_train.values()) > int(ground_dtf.shape[0]/2):
    for_train = int(max(n_train.values()) + remaining / 2)
    tt_ratio = round(1-(for_train/ground_dtf.shape[0]), 6)
else:
    tt_ratio = 0.5

TRAIN, TEST = get_train_test(ground_dtf,
                             n_train,
                             n_test,
                             stratify_on=Y_COL,
                             df_col=X_COLS + [Y_COL],
                             train_test_split_ratio=tt_ratio,
                             method=METHOD_ORDER)


#######################################################################################################################
# Pipeline
#######################################################################################################################
def experiments_pipeline(path, encoder_opt, siamese_options, X_cols, y_col, train, test):
    """
    Pipeline for the experiments. It generates the folder tree, run the methods and produce the table with their
     performance, and finally apply the active learning strategies to the Siamese and Random Forest classifiers
    :param path: Path where the folder tree is generated
    :param encoder_opt: dictionary containing the settings for the encoder
    :param siamese_options: dictionary containing the settings for the Siamese architecture, compiling, and fit
    :param X_cols: column's names for the string pairs
    :param y_col: column's name for the lable
    :param train: dictionaly containing the datasets used for training
    :param test: dictionaly containing the datasets used for testing
    :return: Displays the table of performance and save it in folder 'table'. Then save the performances of the Siamese
    and the Random Forest classifiers with the active learner models in the folder 'active'.
    """
    # 1.1 Generate folder structure
    if not os.path.exists(path):
        os.makedirs(f'{path}models/')
        os.makedirs(f'{path}table/')
        os.makedirs(f'{path}active/')

    # 1.2 Print number of positive and negative samples in training and test sets
    print("Number of positive and negative cases for training and test datasets:")
    for n, df in train.items():
        print((n, df[df['y'] == 1].shape[0], df[df['y'] == 0].shape[0]))
    for n, df in test.items():
        print((n, df[df['y'] == 1].shape[0], df[df['y'] == 0].shape[0]))
    input("Press Enter to continue...")

    # 2. Generate table
    print("Generate table")
    table = get_table(train, test, encoder_opt, siamese_options, X_cols=X_cols, y_col=y_col)
    table.to_csv(f'{path}table/table_performance.csv')
    table.to_pickle(f'{path}table/table_performance.pkl')
    input("Press Enter to continue...")

    # 3. Apply active
    print("Apply active")

    # # # From large training set remove samples already in small training set
    train_large = train[f'large']
    train_small = train[f'small']
    train_large['n1-n2'] = train_large[X_cols[0]] + ' ' + train_large[X_cols[1]]
    train_small['n1-n2'] = train_small[X_cols[0]] + ' ' + train_small[X_cols[1]]
    unlabelled = train_large[~(train_large['n1-n2'].isin(train_small['n1-n2']))]
    unlabelled.reset_index(drop=True, inplace=True)

    key = [k for k in test.keys() if 'R_' in k]
    test_set = {f'R_test': test[k] for k in key}
    apply_active(train_small, unlabelled, test_set, X_cols, y_col)


#######################################################################################################################
# Functions used in pipeline
#######################################################################################################################
# # Compute table performance
def get_table(train, test, encoder_opt, siamese_options, X_cols, y_col):
    """
    Function which generates the table with the performance of the Random Forest, the Siamese, and the 5 Baseline models
    :param train: dictionary with the datasets for training
    :param test: dictionary containing the test datasets
    :param encoder_opt: settings for the padding number and padding character
    :param siamese_options: Siamese settings for the compiler and fitting
    :param X_cols: columns containing company names
    :param y_col: column with the label
    :return: the table with the precision, recall, f1, and support for binary cases, and balanced accuracy, f1 (macro),
    and MCC metrics only for matching cases (y=1)
    """

    # 1. Train models
    # 1.1 Random Forest model
    print("Generate benchmark on train")
    lst_met = ['jaccard', 'jaro', 'fuzz_ratio', 'fuzz_token_ratio', 'distance']
    lst_fet = ['len_parole', 'len_lettere']
    benchmark = benchmark_model_generator(train, X_cols, y_col, lst_met, lst_fet, model_type='rf')

    # 1.2 Siamese model
    print("Generate siamese with distance")
    siamese = siamese_model_generator(train, encoder_opt, siamese_options)

    # 1.3 Deterministic distances with Decision Stump model
    print("Generate baseline on train")
    baseline = {}
    for met in lst_met:
        baseline_per_met = benchmark_model_generator(train, X_cols, y_col, [met], [], model_type='dt', soglia=1)
        baseline.update({f"{met}_{rname}": val for rname, val in baseline_per_met.items()})

    # 2. Compute performance
    print('Compute models performance on tests')
    models_list = [benchmark, siamese, baseline]
    table = models_performance(models_list, test, X_cols, y_col)

    return table

######################################################
# # Apply models with active learners
# # # Benchmark
class BenchmarkWrappedModel(WrappedModel):

    def __init__(self):
        lst_met = ['jaccard', 'jaro', 'fuzz_ratio', 'fuzz_token_ratio', 'distance']
        lst_fet = ['len_parole', 'len_lettere']
        self.model = Benchmark()
        self.model.set_methods(lst_met, lst_fet)

    def full_fit(self, x, y):
        x = pd.DataFrame(x, columns=X_COLS)
        self.__init__()
        self.model.fit(x, y, max_depth_dt=3)  # , n_e=20)

    def batch_fit(self, x_batch, y_batch):
        x_batch = pd.DataFrame(x_batch, columns=X_COLS)
        self.model.fit(x_batch, y_batch, max_depth_dt=3)  # , n_e=20)

    def predict_proba(self, x):
        x = pd.DataFrame(x, columns=X_COLS)
        return self.model.predict_proba(x)


# # # Siamese NN
class SiameseWrappedModel(WrappedModel):

    def __init__(self):
        self.encoder_opt = ENCODER_OPT
        self.siamese_opt = SIAMESE_OPT
        self.batch_name = None
        self.model = Siamese()
        self.model.get_map_encoding(self.encoder_opt)

    def full_fit(self, x, y):
        dtf = pd.DataFrame(x, columns=X_COLS)
        dtf[Y_COL] = y
        self.__init__()
        self.model.fit(dtf, self.siamese_opt)

    def batch_fit(self, x_batch, y_batch):
        dtf = pd.DataFrame(x_batch, columns=X_COLS)
        dtf[Y_COL] = y_batch
        self.model.fit(dtf, self.siamese_opt)

    def predict_proba(self, x):
        dtf = pd.DataFrame(x, columns=X_COLS)
        dtf[Y_COL] = 'dummy_var'
        return self.model.predict_proba(dtf)

    # def predict_uncertainty(self, x):
    #     x = pd.DataFrame(x)
    #     return self.model.predict_uncertainty(x)


# # # Active Learning
def active_plot(name, x_train, x_val, x_test, y_train, y_val, y_test, batch_size, MyWrappedModel,
                fit_mode='full', batch_list=None):
    """
    Function implementing the random and the least-confident learners for the classifier specified in MyWrappedModel
    :param name: name of the classifier, i.e. Siamese or Random Forest
    :param x_train: training dataset containing only features, i.e. company names
    :param x_val: dataset of the unlabeled containing only features, i.e. company names
    :param x_test: test dataset containing only features, i.e. company names
    :param y_train: label of the training dataset
    :param y_val: label of the unlabelled dataset
    :param y_test: label of the test dataset
    :param batch_size: number used for a fixed amount of sample as batch
    :param MyWrappedModel: wrapping class of the classifier
    :param fit_mode: the batch mode updates the classifier fit only using the samples in the batch, the full mode starts
                    a new fit at each step using the labelled + batch samples
    :param batch_list: list specifying the size of the batch at each step
    :return: dataframe with the performances (precision, recall, balanced accuracy, f1_macro) for the learners
    at each step.
    """
    ral = RandomActiveLearner(
        x_train=x_train.values, y_train=y_train.values,
        x_unl=x_val.values, y_unl=y_val.values,
        x_test=x_test.values, y_test=y_test.values,
        batch_size=batch_size, seed=42, model=MyWrappedModel(),
        batch_list=batch_list)

    df_r = ral.train(fit_mode=fit_mode)
    df_r["learner"] = "Random"

    bal = BasicActiveLearner(x_train=x_train.values, y_train=y_train.values,
                             x_unl=x_val.values, y_unl=y_val.values,
                             x_test=x_test.values, y_test=y_test.values,
                             batch_size=batch_size, seed=42, model=MyWrappedModel(),
                             batch_list=batch_list)
    df_b = bal.train(fit_mode=fit_mode)
    df_b["learner"] = "Basic"

    # oal = BoostActiveLearner(x_train=x_train.values, y_train=y_train.values,
    #                         x_unl=x_val.values, y_unl=y_val.values,
    #                         x_test=x_test.values, y_test=y_test.values,
    #                         batch_size=batch_size, seed=42, model=MyWrappedModel(),
    #                         batch_list=batch_list)
    # df_o = oal.train(fit_mode=fit_mode)
    # df_o["learner"] = "Boost"

    df = pd.concat((df_r, df_b)).reset_index(drop=True)
    df["method"] = name
    df.rename(columns={'n_lab': 'N_samples'}, inplace=True)

    return df


def apply_active(train, unlabelled, test_sets, X_cols, y_col):
    """
    Function using active_plot to produce the plots of the balanced accuracy per step for each model
    :param train: training dataset
    :param unlabelled: dataset of the unlabeled
    :param test_sets: test dataset
    :param X_cols: columns of the features, i.e. company names
    :param y_col: column of the label
    :return: save the plots and the dataframe with the performance in the 'active' folder
    """
    batch_list = BATCH_LIST
    batch = BATCH_FIX
    path_save = PATH

    for name, test in test_sets.items():
        x_train = train[X_cols]
        y_train = train[y_col]
        x_val = unlabelled[X_cols]
        y_val = unlabelled[y_col]
        x_test = test[X_cols]
        y_test = test[y_col]

        df_siamese = active_plot('Siamese',
                                 x_train,
                                 x_val,
                                 x_test,
                                 y_train,
                                 y_val,
                                 y_test,
                                 batch_size=batch,
                                 MyWrappedModel=SiameseWrappedModel,
                                 fit_mode='full',
                                 batch_list=batch_list)

        df_bench = active_plot('RandomForest',
                               x_train,
                               x_val,
                               x_test,
                               y_train,
                               y_val,
                               y_test,
                               batch_size=batch,
                               MyWrappedModel=BenchmarkWrappedModel,
                               fit_mode='full',
                               batch_list=batch_list)

        df_siamese['model_learner'] = df_siamese['method'] + '_' + df_siamese['learner']
        df_bench['model_learner'] = df_bench['method'] + '_' + df_bench['learner']
        df_tot = pd.concat([df_siamese, df_bench], ignore_index=True)
        type_sets = {ty: df_tot[df_tot['type_set'] == ty].reset_index(drop=True) for ty in df_tot['type_set'].unique()}

        for df_name, df in type_sets.items():
            df.to_csv(path_save + f'active/Siamese_RandomForest_on_{name}_{df_name}_performance.csv', index=False)

            set_theme()
            plt.figure()
            lineplot(data=df, x="N_samples", y="balanced_accuracy", hue='model_learner')
            plt.title(f'{df_name}: Siamese_RandomForest_AL_on_{name}')
            # plt.show()
            plt.savefig(f'{path_save}active/plot_Siamese_RandomForest_on_{name}_{df_name}_performance.png')


#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == "__main__":
    experiments_pipeline(path=PATH,
                         encoder_opt=ENCODER_OPT,
                         siamese_options=SIAMESE_OPT,
                         X_cols=X_COLS,
                         y_col=Y_COL,
                         train=TRAIN,
                         test=TEST,
                         )
