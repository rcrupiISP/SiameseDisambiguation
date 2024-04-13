# SiameseDisambiguation
Python project for a disambiguation technique based on a Siamese Neural Network and a procedure of Active Learning.

> ### Authors & contributors:
> Alessandro Basile, Riccardo Crupi, Michele Grasso, Alessandro Mercanti, Daniele Regoli, Simone Scarsi, Shuyi Yang, Andrea Claudio Cosentini

To know more about this research work, please refer to our paper:

- Basile, Alessandro, et al. "Disambiguation of Company names via Deep Recurrent Networks." [ESWA (2024)](https://doi.org/10.1016/j.eswa.2023.122035), [arXiv (2023)](https://arxiv.org/abs/2303.05391).

You can cite this work as:
```
@article{basile2024disambiguation,
  title={Disambiguation of company names via deep recurrent networks},
  author={Basile, Alessandro and Crupi, Riccardo and Grasso, Michele and Mercanti, Alessandro and Regoli, Daniele and Scarsi, Simone and Yang, Shuyi and Cosentini, Andrea Claudio},
  journal={Expert Systems with Applications},
  volume={238},
  pages={122035},
  year={2024},
  publisher={Elsevier}
}
```

## Installation
Create a new environment based on Python 3.8.10 and install the requirements.

Python 3.8.10:
```
pip install -r requirements.txt
```

## Quick start
Create a ''data/'' folder. Configure it in config.py and include the dataset to disambiguate in there. Then run:
```
cd prj/app/core/pipelines/services
python Pipeline_experiments.py
```
the output will the placed in the ''experiments/'' folder.

## Settings
The following is a brief description of the parameters collected in the file prj/app/config/config.py that can be modified by the user:
- **Folder tree structure**
  - *PATH_EXP*: is the path to the main folder containing the input (in ''data/'') and the output (in DATE_FOLDER_NAME). If not set, a new folder ''experiments'' is created in the project directory.
  - *FOLDER_NAME*: is the name of the folder where the outputs are saved.
- **Data**
  - *X_COLS*: column names for the features (pair of strings)
  - *Y_COL*: column name for the label
  - *DATA_OPT*: dictionary with the settings for data reading and production. DATA_OPT[''source''] is the name of a file stored in ''data/''
  - *n_train*: dictionary with the name and the size of the training sets, in the form {name: size}
  - *METHOD_ORDER*: the name of a similarity metric: 'jaro' (Jaro-Winkler), 'fuzz_ratio' (InDel ratio), 'fuzz_token_ratio' (token set ratio), 'jaccard', 'distance' (Levensthein).
  - *n_test*: dictionary with the name and the size of test sets in the form {name: [R_size, M_size]}, where R_size is the size of a randomly sampled test set (R_) and M_size is the size of a test set (M_) obtained by using the similarity metric METHOD_ORDER to select the most challenging cases form R_.
- **Neural Network settings**
  - *ENCODER_OPT*: labelencoder settings specifying the maximum length of each string and the padding character used
  - *SIAMESE_OPT*: settings for the compiling and fitting of the siamese neural network
- **Active learning**
  - *BATCH_LIST*: list of batch sizes to use at each step during the active learning
  - *BATCH_FIX*: a fixed size for the batch, used if BATCH_LIST=None

## Input
The input data consist of a pair of strings, being names of real-world entities, and a binary target variable: we use the label 1 for the *matching names* (i.e. when the pair of names corresponds to the same entity), and label 0 for the *non-matching names* (i.e. when the pair of names refers to different entities).
### Get the data
The user can provide the data in two different ways:
- directly as a **pandas DataFrame with 3 columns**, two columns for the pair of names (X=[name_A, name_B]) and one for the label (y), and stored in ''PATH_EXP/data/''. The name of the file (file_name) and its extension (file_ext) has to be setted in DATA_OPT as DATA_OPT[''source'']=''file_name.file_ext''. 
- as a **pandas DataFrame with only the 2 columns** for the pair of names (X) **and containing only matching names**. In this case, the function get_dataset() will generate the non-matching names by randomly pairing the names in the two columns X_COLS. The proportion between matching and non-matching names samples can be set by tuning the DATA_OPT[''n_neg''] parameter in config.py, e.g. DATA_OPT[''n_neg'']=4 will produce 4 non-matching for each matching pair. As in the previous case, this dataframe has to be stored in ''PATH_EXP/data/'' and its name passed to DATA_OPT[''source''].

If DATA_OPT[''source'']=None or the file of the dataset can not be found in the ''data/'' folder, a dummy dataset of acronyms with two columns is downloaded from ''[https://github.com/krishnakt031990/Crawl-Wiki-For-Acronyms/blob/master/AcronymsFile.csv](https://github.com/krishnakt031990/Crawl-Wiki-For-Acronyms/blob/master/AcronymsFile.csv)'' and processed as described above.

Note: allowed files extensions are pickle (.pkl), CSV (.csv with sep=',' or sep=';'), text (.txt with sep='\n'), excel (.xlsx requires openpyxl engine), parquet (.parquet requires pyarrow engine).

### Training and test sets
The samples are then split into two groups for training and testing, taking care of preserving the matching/non-matching ratio. 
By following the description in our paper, we generate several training datasets with different sizes specified in the ''n_train'' variable. These are obtained by hierarchical subsampling elements from the larger sample, i.e. from the training samples, select N_1 samples to form train_1 dataset, from train_1 dataset select N_2 samples (with N_1 > N_2) to form the train_2 dataset, etc. For the following, we assume that there are only three training sets $S^S_{train} \subset S^M_{train} \subset S^L_{train}$.
The sizes of the test sets are specified in the ''n_test'' variable in the form {'test_name': [R_size, M_size]}. Two types of test sets are generated for each ''test_name'': the ''R_'' obtained by randomly selecting R_size elements from the test samples, and the ''M_'' obtained by ordering the ''R_'' samples by a given similarity metric to take the (M_size/2) most dissimilar matching names and the (M_size/2) most similar non-matching names.

The training and test datasets are then collected in the two dictionaries **TRAIN** = \{train_n: $S^n_{train}$, ...\} and **TEST** = \{test_n: $S^{n}_{test}$, ...\}

## How to run the experiments
The procedure to run the experiments is the following:
1. in config.py set:
  - PATH_EXP, with the path to the main folder of the project, containing the folder ''data/'' with the input
  - FOLDER_NAME, with the name of the folder where the output will be saved
  - X_COLS and Y_COL, with the name of the columns for the two features and the label
  - DATA_OPT[''source''], with the name of the input file contained in ''data/''. 
  - n_train and n_test, with the names and the sizes for the training and test sets. The sum between the sizes for the larger training and the larger test must be smaller than the number of all samples.
  - BATCH_LIST or BATCH_FIX, in such a way that it spans all unlabeled samples
2. run Pipeline_experiments.py

These are the minimal actions required to run the experiments with input provided by the user. However, a synthetic dataset can be used by simply setting DATA_OPT[''source'']=None

**Experiments output are under ''DATE_FOLDER_NAME/'':**
- in ''DATE_FOLDER_NAME/table/'', the table with the model's performance in standard supervised classification is saved with the name ```table_performance```
- in ''DATE_FOLDER_NAME/active/'', four plots of the model's performance in the active learning settings are saved.

## Workflow
The project workflow is implemented in **```experiments_pipeline(path, encoder_opt, siamese_options, X_cols, y_col, train, test)```**, and it consists of two different parts:
1. standard supervised classification
2. Active Learning

### Models performance in standard supervised classification
Three different types of classifiers are considered:  
- *baseline models*, i.e. Decision Stups using a string similarity metric as a single feature
- *Random Forest*, using as features several string similarity metrics plus the number of words and the number of characters per string. 
- *Siamese neural network*, a Neural Network classifier on top of a learned LSTM embedding space of strings (for more information refer to our paper)

The performances of the models change as they are trained on training sets of different sizes. Each of the training sets is used to train each of the following 7 models:
- 5 baseline models, one for each of the following metrics: Levenstein distance (Lev), InDel ratio (ID), token set ratio (TSR), Jaccard similarity (Jac), and JW similarity (JW)
- a Random Forest with Lev, ID, TSR, Jac, JW, and number of words and the number of characters per string as features and hyperparameters ```max_depth=3```, ```n_estimators=100```, and ```class_weight=‘balanced’```
- a Siamese neural network, with parameters specified in ENCODER_OPT and SIAMESE_OPT

The method **```get_table(train, test, encoder_opt, siamese_opt, X_cols, y_col)```** generates and trains the 7 models on the training sets specified in **train**. Later the method tests the models to compute precision, recall, f1 score, balanced accuracy, macro-averaged f1, and Matthew's correlation coefficient per each dataset in **test**. 
The output is the table containing the model performances in each test set, which is later stored in the ''DATE_FOLDER_NAME/table/'' folder.

### Models performance in the active learning settings
Active learning procedures are utilized to lower the costs of labeling while maximizing the classifier performances. By starting with a limited number of labeled samples $X_l$ and a set of not-labeled-yet instances $X_u$, active learning considers labeling just a limited number of samples $X_c \subset X_u$ according to some query strategies in a multi-step procedure. Here we adopt an uncertainty sampling strategy where, at each step, instances of $X_u$ on which the prediction of the most updated model is less certain are selected. We denote this strategy as the **least-confident learner**.
These data points are then removed from $X_u$, labeled, and then added to $X_l$ to train a (hopefully) improved version of the classifier. 
![AL](https://user-images.githubusercontent.com/92302358/221153113-e2765603-62a9-4b6e-9f0b-4d2a41c54165.PNG)

To benchmark this procedure, besides the  least-confident learner described above, we run the same experiment with a **random learner**, in which the next-to-be labeled instances are picked at random. 
We assume that the initially labeled instances are the ones in the smaller training set $X^0_l=S^S_{train}$, while the total unlabeled instances are those in the difference between the larger and the smaller training sets, i.e. $X^0_u=S^L_{train}$\ $S^S_{train}$.
At the end of each iteration $j$, the classifier's performance is evaluated as follows: 
1. *pre-train batch test*: test the model $\mathcal{C}^{j}$ on the next-to-be-labelled instances, i.e. $X^{j}_{c}$
2. The model is trained with respect to the new training set $(X^{j}_l, y^{j}_l)$, thus obtaining $\mathcal{C}^{j+1}$
3. $\mathcal{C}^{j+1}$ is tested on:
  - the batch samples $X^{j}_{c}$, and we refer to it as the *post-train batch test*
  - the updated unlabelled set $X^{j}_u$, i.e. all the remaining unlabelled instances, and we refer to it as the *not-labeled-yet test*
  - the actual *test sets*

The method **```apply_active(train_small, unlabelled, test_set, X_cols, y_col)```** implements the active learning settings just described to train a Random Forest and a Siamese Neural Network. 
At each step, the method computes precision, recall, f1 score, balanced accuracy, and macro-averaged f1 for the four types of tests **pre-train batch test**, **post-train batch test**, **not-labeled-yet test**, and **test set**. 
The outputs are the tables containing the classifiers' performance at each step and the plots of the balanced accuracy for the four types of tests. The outputs are  stored in the ''DATE_FOLDER_NAME/active/'' folder.
