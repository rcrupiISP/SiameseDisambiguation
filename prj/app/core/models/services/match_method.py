from prj.app.core.data.services.utils_string_cleaning import string_cleaning
from fuzzywuzzy import fuzz
import jaro
import pandas as pd
from Levenshtein import distance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from numpy.random import seed
seed(16)

class Benchmark:
    def __init__(self):
        self.columns = None
        self.clf = None
        self.lst_str_match = []
        self.lst_str_feat = []
        self.lst_col_match = None
        self.bln_clean_str = None

    @staticmethod
    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    @staticmethod
    def len_parole(list):
        return len(list.split(' '))

    @staticmethod
    def len_lettere(list):
        return len(list)

    def set_methods(self, lst_met, lst_fet):
        # String match
        if 'jaccard' in lst_met:
            self.lst_str_match.append(self.jaccard)
        if 'jaro' in lst_met:
            self.lst_str_match.append(jaro.jaro_winkler_metric)
        if 'fuzz_ratio' in lst_met:
            self.lst_str_match.append(fuzz.ratio)
        if 'fuzz_token_ratio' in lst_met:
            self.lst_str_match.append(fuzz.token_set_ratio)
        if 'distance' in lst_met:
            self.lst_str_match.append(distance)
        # Feature string
        if 'len_parole' in lst_fet:
            self.lst_str_feat.append(self.len_parole)
        if 'len_lettere' in lst_fet:
            self.lst_str_feat.append(self.len_lettere)

    def prepare_dataset(self, X, lst_col_match, bln_clean_str):
        lst_str_match = self.lst_str_match
        lst_str_feat = self.lst_str_feat
        lst_col_old = X.columns

        if len(lst_col_match) != 2:
            print("Error, not two columns to be matched.")
            raise

        for met in lst_str_match:
            # print("Extracting match method ", met)
            if bln_clean_str:
                X[met.__name__] = X[lst_col_match].apply(lambda x: met(
                    string_cleaning(x[0]),
                    string_cleaning(x[1])
                ), axis=1)
            else:
                X[met.__name__] = X[lst_col_match].apply(lambda x: met(x[0], x[1]), axis=1)
        for met in lst_str_feat:
            # print("Extracting feature ", met)
            X[met.__name__ + '_' + lst_col_match[0]] = X[lst_col_match[0]].apply(lambda x: met(x))
            X[met.__name__ + '_' + lst_col_match[1]] = X[lst_col_match[1]].apply(lambda x: met(x))

        col_to_train = [i for i in X.columns if i not in lst_col_old]

        return X[col_to_train]

    def fit(self, dtf, y, lst_col_match=None, bln_clean_str=True, max_depth_dt=3, str_model='rf',
            test_size=0, n_e=100):
        """
        Run the matching phase.
        :param dtf: dataset with target and the couple of strings.
        :param y: Target label.
        :param lst_col_match: List of the two columns to be matched. E.g. ['DES_INTESTAZIONE', 'COMPANY_NAME_x'].
        :param bln_clean_str: if true the string is cleaned.
        :param max_depth_dt: max depth of the decision tree.
        :param str_model: 'dt' for decision tree or 'rf' for random forest.
        :param test_size: float, percentage of the validation set to test the algorithm.
        :param n_e: Number of the estimator for Random Forest.
        :return: prediction match per each row.
        """
        seed(16)

        if lst_col_match is None:
            lst_col_match = dtf.columns
            if len(lst_col_match) != 2:
                print("The columns are not two. Two column string needed.")
                raise

        self.lst_col_match, self.bln_clean_str = lst_col_match, bln_clean_str
        # Build dataset
        X = dtf.copy()
        X = self.prepare_dataset(X, lst_col_match, bln_clean_str)

        # Train and test dataset
        if test_size != 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size, random_state=42)
        else:
            X_train, _, y_train, _ = train_test_split(X, y, test_size=1, random_state=42)
            X_test = None
            y_test = None

        self.columns = X_train.columns
        if str_model == 'dt':
            clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth_dt)
        elif str_model == 'rf':
            clf = RandomForestClassifier(random_state=0, max_depth=max_depth_dt,
                                         n_estimators=n_e, class_weight="balanced")
        else:
            print("Classifier model not specified. Decision Tree chosen.")
            clf = DecisionTreeClassifier(random_state=0, max_depth=max_depth_dt)
        # print(X_train.columns) # Inserted to extract features' names
        clf.fit(X_train, y_train)

        # if str_model == 'dt':
        #     text_representation = tree.export_text(clf, feature_names=list(X_train.columns))
        #     print(text_representation)
        # if str_model == 'rf':
        #     print(pd.DataFrame((zip(X_train.columns, clf.feature_importances_)), columns=['col', 'FI']).sort_values(
        #         by=['FI'], ascending=False))
        self.clf = clf

        # Prediction and performance
        if X_test is None or y_test is None:
            print("Training performance.")
            y_pred = clf.predict(X_train)
            print(classification_report(y_train, y_pred))
        else:
            print("Validation performance.")
            y_pred = clf.predict(X_test)
            print(classification_report(y_test, y_pred))

    def predict(self, dtf):
        X = dtf.copy()
        X = self.prepare_dataset(X, self.lst_col_match, self.bln_clean_str)
        return self.clf.predict(X)

    def predict_proba(self, dtf):
        X = dtf.copy()
        X = self.prepare_dataset(X, self.lst_col_match, self.bln_clean_str)
        return self.clf.predict_proba(X)

    def predict_proba_one(self, dtf):
        X = dtf.copy()
        X = self.prepare_dataset(X, self.lst_col_match, self.bln_clean_str)
        return self.clf.predict_proba(X)[:, 1]

    def plot_dt(self, figsize=(15, 10)):
        try:
            fig = plt.figure(figsize=figsize)
            _ = tree.plot_tree(self.clf, feature_names=self.columns, filled=True)
            plt.tight_layout()
            plt.show()
        except:
            plt.close(fig)
            print("Not a decision Tree.")
