##############################################################################
# Script for generating the plots in the active learning experiments

import pandas as pd
import matplotlib.ticker as tick_form
import matplotlib.pyplot as plt
from seaborn import lineplot, set_theme

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

main_folder = 'example'

def cmb_pt(cm, set):

    experiment_folder = f"2023-02-24_acronym_{cm}/"
    active_file_name = f"active/Siamese_RandomForest_on_R_test_{set}.pkl"
    return main_folder + experiment_folder + active_file_name


for set in ["post_batch_performance", "pre_batch_performance", "test_performance", "unl_performance"]:
    AB_C = pd.read_pickle(cmb_pt("AB-C", set))
    AC_B = pd.read_pickle(cmb_pt("AC-B", set))
    BC_A = pd.read_pickle(cmb_pt("BC-A", set))
    for df, comb in zip([AB_C, AC_B, BC_A], ['AB_C', 'AC_B', 'BC_A']):
        df['model_learner_comb'] = df['model_learner'] + '_' + comb
        df['model_learner_rnm'] = df['model_learner']  # .map({'Siamese_Random'})
        df['Model-Learner'] = df['model_learner'].map({'Siamese_Random': 'Siamese-Random',
                                                       'Siamese_Basic': 'Siamese-Least Confident',
                                                       'Benchmark_Random': 'RandomForest-Random',
                                                       'Benchmark_Basic': 'RandomForest-Least Confident'})
    df_tot = pd.concat([AB_C[['N_samples', 'balanced_accuracy', 'Model-Learner', 'method']],
                        AC_B[['N_samples', 'balanced_accuracy', 'Model-Learner', 'method']],
                        BC_A[['N_samples', 'balanced_accuracy', 'Model-Learner', 'method']]], ignore_index=True)
    set_theme()

    fig, ax = plt.subplots()
    h = 'Model-Learner'
    lineplot(data=df_tot, x="N_samples", y="balanced_accuracy", hue=h, ci=95, style=h, dashes=False,
             markers=["o", "o", "o", "o"], ax=ax)
    # plt.title(set)
    plt.legend(loc='lower right')
    plt.ylabel('Balanced Accuracy')
    plt.xlabel('Number of labelled samples')
    plt.savefig(
        f'{main_folder}active_experiments/active_{set}_ci95.png')

    ax.set_xscale('log')
    ax.set_xticks([100, 200, 400, 800, 1600, 3200, 6000])
    plt.xticks(rotation=45)
    ax.get_xaxis().set_major_formatter(tick_form.ScalarFormatter())
    plt.savefig(
        f'{main_folder}/active_experiments/active_{set}_ci95_log.png')
