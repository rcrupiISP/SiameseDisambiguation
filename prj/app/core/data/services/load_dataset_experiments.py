import pandas as pd
import os
from random import choices
from datetime import datetime
import requests
from pandas.errors import ParserError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from prj.app.core.models.services.match_method import Benchmark
from prj.app.config.config import SOURCE_DEFAULT


def split_dtf(dtf_in, split, level):
    """
    Split the dataframe in two complementary parts. It is a customised version of the sklearn 'train_test_split()'
    :param level: column used to stratify on
    :param dtf_in: dataframe to split
    :param split: fraction used to split the dtf
    :return: the splitted dataframes
    """
    dtf = dtf_in.copy()

    # In case frac = 1, return the same dataframe
    frac_size = split / dtf.shape[0]
    if round(frac_size, 2) >= 0.99:
        df_samp = dtf.sample(frac=frac_size, replace=True, random_state=0)
        return dtf, df_samp.reset_index(drop=True)

    if type(split) == float:
        dist = dtf[level]
        larger_ex, smaller_ex = train_test_split(dtf,
                                                 test_size=split,
                                                 random_state=7,
                                                 stratify=dist)
    elif type(split) == int:
        dist = dtf[level]
        larger_ex, smaller_ex = train_test_split(dtf,
                                                 test_size=frac_size,
                                                 random_state=7,
                                                 stratify=dist)
    else:
        larger_ex = dtf.copy()
        smaller_ex = dtf.copy()
    return larger_ex.reset_index(drop=True), smaller_ex.reset_index(drop=True)


def swap_colunms(dtf, col_swap, portion):
    """
    Function used to swap randomly the entries of two columns
    :param dtf: dataframe
    :param col_swap: list of the two columns to swap
    :param portion: fraction of entries to swap
    :return: a dataframe with portion*dtf.shape[0] entries swapped between the two columns col_swap
    """
    if portion == 1:
        df_res = pd.concat([dtf[col_swap], dtf[[col_swap[1], col_swap[0]]]], ignore_index=True)
    else:
        df_res = dtf.copy()
        swap_index = df_res.sample(frac=portion, random_state=0).index
        df_res.loc[swap_index, col_swap[0]] = dtf.loc[swap_index, col_swap[1]]
        df_res.loc[swap_index, col_swap[1]] = dtf.loc[swap_index, col_swap[0]]
    return df_res.reset_index(drop=True)


def keep_n_neg(gr, wd_2, n_neg, col):
    """
    function applied to a grouped datarame and used to have n_neg couple (negative matches) per each group (indexed by
    a company name in df_col[0])
    :param gr: group element of the grouped dataframe
    :param wd_2: list of string (company names) to choose from
    :param n_neg: number of strings to choose
    :param col: column name
    :return: a group element having n_neg entries from wd_2 and index a company name in df_col[0],
                in the form: index = 'name_A', data = [n_neg other names]
    """
    wd1 = set(gr[col])
    rem = list(wd_2 - wd1)
    gc = pd.DataFrame({col: choices(rem, k=n_neg)})
    return gc


def append_negatives(dtf, n_neg, df_col):
    """
    From a datafrem of matching string couples, for each string in one column produce n_neg non-matching couples
    :param dtf: datafrem of matching string couples
    :param n_neg: number of non-matching couples per match
    :param df_col: column names
    :return: dataframe with dtf.shape[0] matching couples and n_neg*dtf.shape[0] non-matching couples
    """
    print(f"Find negative matches:{datetime.now()}")
    neg = set(dtf[df_col[1]].unique())
    dtf_pos = dtf.copy()
    gr_neg = dtf_pos.groupby(df_col[0])
    pos_len = len(gr_neg)  # to ensure the proportion n_neg negatives/each positive, take only pos_len positive cases
    dtf_neg = gr_neg.apply(lambda gr: keep_n_neg(gr, neg, n_neg, df_col[1]))
    dtf_neg.reset_index(inplace=True)
    print(f"{datetime.now()}: Found {dtf_neg.shape[0]} negative cases")

    dtf_pos = dtf_pos.sample(n=pos_len, random_state=8)
    dtf_pos[df_col[2]] = 1
    dtf_neg[df_col[2]] = 0
    dtf_return = pd.concat([dtf_neg[df_col], dtf_pos[df_col]], ignore_index=True)
    return dtf_return.sample(frac=1, random_state=0).reset_index(drop=True)


def read_file(file_exp):
    """
    Open some of the most used file types in pandas: pkl, csv, xlsx, parquet. Note that requires specific engine
    installed to work properly
    :param file_exp: path to file
    :return: the dataframe read from file_exp
    """
    file_name = file_exp.split('/')[-1]
    file_type = {'pkl': pd.read_pickle,
                 'csv': pd.read_csv,
                 'txt': lambda x: pd.read_csv(x, sep='\n'),
                 'xlsx': lambda x: pd.read_excel(x, engine='openpyxl'),  # TODO: requires openpyxl to be installed,
                 'parquet': lambda x: pd.read_parquet(x, engine='pyarrow')  # TODO: requires pyarrow to be installed
                 }
    extension = file_name.split('.')[-1]
    try:
        df = file_type.get(extension)(file_exp)
    except ParserError:
        df = file_type.get(extension)(file_exp, sep=';')
    return df


def get_dataset(path, frac, n_neg, df_col, source=SOURCE_DEFAULT):
    """
    According to source, load a datarfame from file or get the one in
    url:https://raw.githubusercontent.com/krishnakt031990/Crawl-Wiki-For-Acronyms/master/AcronymsFile.csv.
    Then, if the loaded dtf has 3 columns, assume the first 2 are the features (string names) and the third is the lable.
    If the loaded dtf has 2 columns, assume they are couples of matching string names and append n_neg non-matching
    per each matching couples
    :param path: path to file
    :param frac: if generate from positive samp., shuffle frac% of the dtf
    :param n_neg: if generate from positive samp., append n_neg non-matchings per matching couple
    :param df_col: column names
    :param source: source can be a file name (with its extension). If not, it try to import from url the example
    :return: the dataframe of all samples, later used to produce training and test sets. It has 3 columns with
    df_col[:-1] columns name for features and df_col[2] column name used for the label
    """
    file_exp = f"{path}{source}"
    if os.path.exists(file_exp):
        print(f'{datetime.now()}: Read file {source}')
        df = read_file(file_exp)
    else:
        print(f'{datetime.now()}: Get file from url {SOURCE_DEFAULT}')
        response = requests.get(SOURCE_DEFAULT, verify=False)
        lines = (response.text).split('\n')
        lst_couple = [[i.split(' - ', 1)[0], i.split(' - ', 1)[1]] for i in lines if len(i.split(' - ', 1)) == 2]
        df = pd.DataFrame(lst_couple, columns=df_col[:-1])

    col = df.columns
    assert (1 < len(col) < 4), f"Two or three columns are expected, got {len(col)}: {list(df.columns)}"

    if len(col) == 3:
        df.rename(columns={col[i]: c for i, c in enumerate(df_col)}, inplace=True)
        for c in df_col[: -1]:
            df[c] = df[c].str.upper()
            df[c] = df[c].apply(lambda x: " ".join(x.split()))
        return df
    else:
        print(f'{datetime.now()}: Generate negatives')
        for i, c in enumerate(col):
            df[df_col[i]] = df[c].str.upper()
            df[df_col[i]] = df[df_col[i]].apply(lambda x: " ".join(str(x).split()))
        dtf_positives = df[df_col[:-1]].drop_duplicates()
        dtf_ret = swap_colunms(dtf_positives.loc[:, df_col[:-1]], df_col[:-1], portion=frac)

        neg_pos_dtf = append_negatives(dtf_ret, n_neg, df_col)
        return neg_pos_dtf


def apply_method(dtf_in, n_diff, method, df_col):
    """
    Order the samples in dtf_in according to a similarity method and then select the n_diff/2 positive with lower
    similarity (less similar matching cases) and n_diff/2 negative with higher similarity (more similar non-matching
     cases)
    :param dtf_in: dataframe with matching and non-matching couples
    :param n_diff: size of samples to choose
    :param method: similarity score used to order the samples
    :param df_col: column names
    :return: dataset of n_diff/2 less similar matching cases and n_diff/2 more similar non-matching cases
    """
    df_pos = dtf_in[dtf_in[df_col[2]] == 1].copy()
    df_neg = dtf_in[dtf_in[df_col[2]] == 0].copy()
    met_bench = Benchmark()
    met_bench.set_methods(method, [])

    df_pos[method] = met_bench.prepare_dataset(df_pos.copy(), df_col[:-1], bln_clean_str=True)
    df_neg[method] = met_bench.prepare_dataset(df_neg.copy(), df_col[:-1], bln_clean_str=True)
    # NOTE: here we use similarity measure, i.e. the highest, the more similar!
    pos = df_pos.sort_values(method).head(int(n_diff / 2))
    neg = df_neg.sort_values(method, ascending=False).head(int(n_diff / 2))
    plt.figure()
    plt.hist([pos[method], neg[method]], label=[f'positive_{method}', f'negative_{method}'])
    plt.legend(loc='upper left')
    plt.show()
    dtf_out = pd.concat([pos[df_col], neg[df_col]])
    return dtf_out.sample(frac=1, random_state=8).reset_index(drop=True)


def subsample_train(sample_dtf, n_split, y_col):
    """
    Function that hierarchically split the training sets starting from sample_dtf
    :param sample_dtf: samples used only for training
    :param n_split: number of samples per training dtf
    :param y_col: column of label used to stratify the splitting
    :return: training sets with size specified in n_split
    """
    subsample = {}
    for samp, n_samp in sorted(n_split.items(), key=lambda x: x[1], reverse=True):
        print(f'{datetime.now()}: subsampling {samp} for train')
        _, subsample[samp] = split_dtf(sample_dtf, split=int(n_samp), level=y_col)
        sample_dtf = subsample[samp]
    return subsample


def subsample_test(dtf, name, n_split, method, df_col):
    """
    Function that splits dtf in a random-ordered (R_) and a method-ordered (M_) test set
    :param dtf: samples used only for testing
    :param name: name of the test
    :param n_split: list of sizes for the R_ and the M_ sets
    :param method: similarity method used to order
    :param df_col: column names
    :return: two datasets, one R_ from dtf, and one M_ from the R_ set
    """
    acronim = {'jaro': 'JW', 'fuzz_ratio': 'InDel', 'fuzz_token_ratio': 'TSR', 'jaccard': 'JACC', 'distance': 'LEV'}
    met = acronim.get(method, 'undefined')
    _, simp = split_dtf(dtf, split=int(n_split[0]), level=df_col[2])
    diff = apply_method(simp, n_split[1], method, df_col)
    return {f'R_{name}': simp, f'{met}_{name}': diff}


def get_train_test(ground_dtf, n_train, n_test, df_col, stratify_on, train_test_split_ratio, method):
    """
    Generate train and tests from ground_dtf. Training sets are obtained by hierarchical subsampling.
    Tests are of two types: sampled from ground_dtf and ordered by similarity metric
    :param df_col: dataframe columns X_COLS + [Y_COL]
    :param ground_dtf: database with all samples
    :param n_train: dictionary of training size {name_train: numb_of_samples}
    :param n_test: dictionary of test size {name_test: [numb_of_samples, numb_of_ordered_samples]}
    :param stratify_on: lable to stratify on
    :param train_test_split_ratio: ratio used to split between training and test
    :param method: method used to order the test
    :return: two dictionary ---one for training sets and one for test sets--- with values 'name_dt': dataset.
            The keys are the same as n_train, n_test.
    """
    train_part, test_part = split_dtf(ground_dtf, train_test_split_ratio, level=stratify_on)

    train = subsample_train(train_part, n_train, df_col[2])
    test = dict()
    for key, size_list in n_test.items():
        per_test = subsample_test(test_part, key, size_list, method, df_col)
        test.update(per_test)

    return train, test
