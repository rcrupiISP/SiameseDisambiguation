# Example: acronyms disambiguation

We propose the application of the methods in prj to perform acronyms disambiguation.

## Data
The experimental data are couples of strings composed by full expressions and their corresponging acronyms, extracted from  ''[https://github.com/krishnakt031990/Crawl-Wiki-For-Acronyms/blob/master/AcronymsFile.csv](https://github.com/krishnakt031990/Crawl-Wiki-For-Acronyms/blob/master/AcronymsFile.csv)''.
These matched pairs are imported into a two-column dataframe, to which non-matching names are added by randomly pairing acronyms and full expressions. The proportion between matching and non-matching couples is set to have 4 non-matching for each matching pair (DATA_OPT[''n_neg'']=4 in **```get_dataset()```** module). 

To generalise the results, we apply a stratified three-fold cross-validation technique: by taking three different samples of 3000 pairs each ---600 matching, 2400 non-matching pairs--- (stored in ''example/data/samples/''), iteratively we choose one of them as test set and keep the union of the rest as a training set. Nameing the samples as *samp_A*, *samp_B*, and *samp_C*, we end up with the three combinations:
- *AB-C*, where  $S^L_{train}$ = *samp_A* $\cup$ *samp_B* and $S^R_{test}$ = *samp_C*
- *AC-B*, where  $S^L_{train}$ = *samp_A* $\cup$ *samp_C* and $S^R_{test}$ = *samp_B*
- *BC-A*, where  $S^L_{train}$ = *samp_B* $\cup$ *samp_C* and $S^R_{test}$ = *samp_A*

For each combination, from $S^L_{train}$ we subsample other two training sets, medium-sized training set $S^M_{train}$ of 2000 couples and small-sized training set $S^S_{train}$ of 100 couples, while from $S^R_{test}$ we subsample $S^{JW}_{test}$ by taking the 200 most challenging couples according to the Jaro-Winkler similarity. The training and test sets for each ```combination``` (*AB-C*, *AC-B*, and *BC-A*) are stored in ''example/data/```combination```/''. 

The pipeline implementing the supervised learning and the active learning setups **```experiments_pipeline(path, encoder_opt, siamese_options, X_cols, y_col, train, test)```** is run for each combination with the same parameters, and the results are saved in ''example/2023-02-24_acronym_```combination```/''.

Finally the single-combination outputs are aggregated to obtain the following cross-validated results saved in ''example/outputs/'':
- **table_BA_F1_MCC.csv**: the mean and the standard deviation of the balanced accuracy, f1 macro-averaged, and MCC for each classifier in each test $S^R_{test}$ and $S^{JW}_{test}$
- **table_prec_rec_f1.csv**: the mean and the standard deviation of the precision, recall and f1 score for each classifier in each test $S^R_{test}$ and $S^{JW}_{test}$
- **active_```set```_performance_ci95_log.png**: plot of the balanced accuracy computed in the active learning setup. The plots are in a logarithmic scale over the x axis and the mean and 95% normal confidence intervals are obtained by aggregating the balanced accuracy over the 3 cross validation folds. The ```set``` refers to the different type of samples on which the classifier is tested, according to the following steps:
  1. *pre_batch*: at the $j^{th}$ iteration, test the classifier $\mathcal{C}^{j}$ on the next-to-be-labelled instances, i.e. $X^{j}_{c}$
  2. The classifier is trained with respect to the new training set, thus obtaining $\mathcal{C}^{j+1}$
  3. $\mathcal{C}^{j+1}$ is tested on:
    - the batch samples $X^{j}_{c}$, and we refer to it as the *post_train*
    - the updated unlabelled set $X^{j}_u$, i.e. all the remaining unlabelled instances, and we refer to it as *unl*
    - the actual test set *test*.
    
