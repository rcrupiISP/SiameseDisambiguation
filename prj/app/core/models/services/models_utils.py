import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, f1_score, matthews_corrcoef
from prj.app.core.models.services.match_method import Benchmark
from prj.app.core.models.services.LSTM_class import Siamese


def benchmark_model_generator(dtf_train, X_cols, y_col, lst_met, lst_fet, model_type='rf', soglia=3):
    """
    Uses the class Benchmark to generate a binary classifier based on lst_met and lst_fet features.
    :param dtf_train: dataset used for training
    :param X_cols: features column names
    :param y_col: label column name
    :param lst_met: list with the similarity method used as features
    :param lst_fet: additional features, i.e. number of words and number of characters (length of the string)
    :param model_type: 'rf' to initialise a RandomForest, or 'dt' to initialise a DecisionTree
    :param soglia: maximum dept of the tree
    :return: a classifier (RF or DT) trained on dtf_train
    """
    fitted_models = dict()
    classifier_type = {'rf': 'RF', 'dt': ('Dec_stump' if soglia == 1 else 'Dec_tree')}

    for name, train in dtf_train.items():
        print(f'Fit {lst_met} with {name}')
        met_bench = Benchmark()
        met_bench.set_methods(lst_met, lst_fet)
        met_bench.fit(train[X_cols], train[y_col],
                      bln_clean_str=True,
                      max_depth_dt=soglia,
                      str_model=model_type,
                      test_size=0)

        fitted_models[f"{classifier_type.get(model_type, 'benchmark')}_{name}"] = met_bench
    return fitted_models

def siamese_model_generator(dtf_train, encoder_opt, siamese_opt):
    """
    Uses the LSTM_class to generate a binary classifier based on a Siamese neural network
    :param dtf_train: training dataset
    :param encoder_opt: settings for the LabelEncoder (padding length, padding char)
    :param siamese_opt: settings for the compiling and fitting of the neural network
    :return: a classifier (Siamese NN) trained on dtf_train
    """
    fitted_models = dict()
    for name, train in dtf_train.items():
        print(f'Fit siamese with {name}')
        siamese = Siamese()
        siamese.get_map_encoding(encoder_opt)
        siamese.fit(train, fit_options=siamese_opt)
        fitted_models[f"siamese_{name}"] = siamese
    return fitted_models

def model_predict_per_test(models, dtf, X_cols, y_col):
    pred = {}
    for mod_name, mod in models.items():
        print(mod_name)
        if 'siamese' in mod_name:
            pred[mod_name] = [dtf[y_col], mod.predict(dtf)]
        else:
            pred[mod_name] = [dtf[y_col], mod.predict(dtf[X_cols])]
    return pred

def models_performance(models_set, test_set, X_cols, y_col):
    """
    For each classifier in models_set compute the 'Precision', 'Recall', 'F1-score', 'Support',
    'Balanced_accuracy', 'F1' (macro), 'MCC' metrics over the tests in test_set
    :param models_set: list of classifiers
    :param test_set: collection of datasets used for testing
    :param X_cols: column names of the features, i.e. columns containing the string names
    :param y_col: column name for label
    :return: table of performance for each model in each test
    """
    final_df = []

    #for df_name, dtf in test_set.items():
    for df_name, dtf in sorted(test_set.items(), key=lambda x: x[1].shape[0], reverse=True):
        pred = {}
        for models in models_set:
            pred_per_model = model_predict_per_test(models, dtf, X_cols, y_col)
            pred.update(pred_per_model)

        report = []
        mux = pd.MultiIndex.from_product([[df_name],
                                          ['Precision', 'Recall', 'F1-score', 'Support', 'Balanced_accuracy', 'F1', 'MCC']],
                                         names=['test_set', 'metrics'])
        for mod_name, y_pd in pred.items():
            df_perf = pd.DataFrame(list(precision_recall_fscore_support(y_true=y_pd[0], y_pred=y_pd[1]))+
                           [np.array(['_', round(balanced_accuracy_score(y_pd[0], y_pd[1]), 6)])]+
                           [np.array(['_', round(f1_score(y_pd[0], y_pd[1]), 6)])]+
                           [np.array(['_', round(matthews_corrcoef(y_pd[0], y_pd[1]), 6)])],
                              index=mux).T
            df_perf['model'] = mod_name
            report = report + [df_perf]
        df_model = pd.concat(report)
        df_model = df_model.reset_index().set_index(['model', 'index'])
        df_model = df_model.rename_axis(index=['model', 'cases'])
        final_df = final_df + [df_model]
    final_report = pd.concat(final_df, axis=1)
    print(final_report)
    return final_report


def confusion_matrix_among_models(models_to_compare, dtf, X_cols, y_col):
    """Compute the confusion matrix between two model
    :param models_to_compare: dictionary with the two models we want to compare
    :param dtf: test dataset
    :param X_cols: column names containing features
    :param y_col: column name containing label
    :return: print and return a datset with the confusion matrix among the two model
    """
    pred= model_predict_per_test(models_to_compare, dtf, X_cols, y_col)
    name = list(models_to_compare.keys())
    data_in = {y_name: y_pred[1] for y_name, y_pred in pred.items()}
    data_in['y'] = pred[name[0]][0]
    among_models = pd.DataFrame(data_in)
    print(f'Confusion matrix among {name[0]} and {name[1]}:')
    cm_00 = among_models[(among_models[name[0]]==0)&(among_models[name[1]]==0)&(among_models['y']==0)].shape[0]
    cm_01 = among_models[(among_models[name[0]]==0)&(among_models[name[1]]==0)&(among_models['y']==1)].shape[0]
    cm_10 = among_models[(among_models[name[0]]==0)&(among_models[name[1]]==1)&(among_models['y']==0)].shape[0]
    cm_11 = among_models[(among_models[name[0]]==0)&(among_models[name[1]]==1)&(among_models['y']==1)].shape[0]
    cm_20 = among_models[(among_models[name[0]]==1)&(among_models[name[1]]==0)&(among_models['y']==0)].shape[0]
    cm_21 = among_models[(among_models[name[0]]==1)&(among_models[name[1]]==0)&(among_models['y']==1)].shape[0]
    cm_30 = among_models[(among_models[name[0]]==1)&(among_models[name[1]]==1)&(among_models['y']==0)].shape[0]
    cm_31 = among_models[(among_models[name[0]]==1)&(among_models[name[1]]==1)&(among_models['y']==1)].shape[0]
    print(f'{name[0]}/{name[1]}|\t0\t|\t1\t|')

    print(f'0/0|\t{cm_00}\t|\t{cm_01}\t|')
    print(f'0/1|\t{cm_10}\t|\t{cm_11}\t|')
    print(f'1/0|\t{cm_20}\t|\t{cm_21}\t|')
    print(f'1/1|\t{cm_30}\t|\t{cm_31}\t|')

    return pd.DataFrame(data=[['0/0', cm_00, cm_01],
                              ['0/1', cm_10, cm_11],
                              ['1/0', cm_20, cm_21],
                              ['1/1', cm_30, cm_31]],
                        columns=['test set', f'{name[0]}/{name[1]}', '0', '1'])