# SiameseDisambiguation
Python project for a disambiguation technique based on a Siamese Neural Network and a procedure of Active Learning.

> ### Authors & contributors:
> Alessandro Basile, Riccardo Crupi, Michele Grasso, Alessandro Mercanti, Daniele Regoli, Simone Scarsi, Shuyi Yang, Andrea Claudio Cosentini

To know more about this research work, please refer to our paper:

- Disambiguation of Company names via Deep Recurrent Networks


## Installation
Create a new environment based on Python 3.8.10 and install the requirements.

Python 3.8.10:
```
pip install -r requirements.txt
```
## Settings
The following is a brief description of the parameters collected in the file prj/app/config/config.py that can be modified by the user:
- **Folder tree structure**
  - *PATH_EXP*: is the path to the main folder containing the input (in ''data/'') and the output (in DATE_FOLDER_NAME). If not setted, a new folder ''experiments'' is created in the project directory.
  - *FOLDER_NAME*: is the name of the folder where the output are saved.
- **Data**
  - *X_COLS*: column names for the features (pair of strings)
  - *Y_COL*: column name for the label
  - *DATA_OPT*: dictionary with the settings for data reading and production. If DATA_OPT[''source''] a file name (with its extension), it looks for that file in ''data/''
  - *n_train*: dictionary with the name and the size of the training sets, in the form {name: size}
  - *METHOD_ORDER*: the name of a similarity metric: 'jaro' (Jaro-Winkler), 'fuzz_ratio' (InDel ratio), 'fuzz_token_ratio' (token set ratio), 'jaccard', 'distance' (Levensthein).
  - *n_test*: dictionary with the name and the size of test sets in the form {name: [R_size, M_size]}, where R_size is the size of a randomly sampled test set (R_) and M_size is the size of a test sets (M_) obtained by using the similarity metric METHOD_ORDER to select the most challenging cases form R_.
- **Neural Network settings**
  - *ENCODER_OPT*: labelencoder settings specifying the maximum lenght of each string and the padding character used
  - *SIAMESE_OPT*: settings for the compiling and fitting of the siamese neural network
- **Active learning**
  - *BATCH_LIST*: list of batch sizes to use at each step during the active learning
  - *BATCH_FIX*: a fixed size for the batch, used if BATCH_LIST=None

## Input
The input data consist of a pair of strings, being names of real-world entities, and a binary target variable: we use the label 1 for the *matching names* (i.e. when the pair of names corresponds to the same entity), and label 0 for the *non-matching names* (i.e. when the pair of names refers to different entities).
### Get the data
The user can provide the data in two different ways:
- directly as a **pandas DataFrame with 3 columns**, two columns for the pair of names (X=[name_A, name_B]) and one for the lable (y), and stored in ''PATH_EXP/data/''. The name of the file (file_name) and its extension (file_ext) has to be setted in DATA_OPT as DATA_OPT[''source'']=''file_name.file_ext''. 
- as a **pandas DataFrame with only the 2 columns** for the pair of names (X) **and containing only matching names**. In this case, the function get_dataset() will generate the non-matching names by random pairing the names in the two columns X_COLS. The proportion between matching and non-matching names samples can be setted by tuning the DATA_OPT[''n_neg''] parameter in config.py, e.g. DATA_OPT[''n_neg'']=4 will produce 4 non-matching for each matching names. As in the previous case, the two columns dataframe has to be stored in ''PATH_EXP/data/'' and its name passed to DATA_OPT[''source''].

If DATA_OPT[''source'']=None or the file of the dataset can not be found in ''data/'' folder, a dummy dataset of acronyms with two columns is downoaded from ''https://raw.githubusercontent.com/krishnakt031990/Crawl-Wiki-For-Acronyms/master/AcronymsFile.csv'' and processed as described above.

Note: allowed files extensions are pickle (.pkl), CSV (.csv with sep=',' or sep=';'), text (.txt with sep='\n'), excel (.xlsx requires openpyxl engine), parquet (.parquet requires pyarrow engine).

### Training and test sets
The samples are then splitted in two groups for training and for testing, taking care of preserving the matching/non-matching ratio. 
By following the description in our paper, we generate a number of training datasets with differen sizes specified in ''n_train'' variable. These are obtained by hierarchical subsampling elements from the larger sample, i.e. from the training samples, select N_1 samples to form train_1 dataset, from train_1 dataset select N_2 samples (with N_1 > N_2) to form train_2 dataset, etc. In the following we assume of having only three training set $S^S_{train} \subset S^M_{train} \subset S^L_{train}$.
The size of the test sets are specified in the ''n_test'' variable in the form {'test_name': [R_size, M_size]}. Two types of test sets are generated for each ''test_name'': the ''R_'' obtained by randomly selecting R_size elements from the test samples, and the ''M_'' obtained by ordering the ''R_'' samples by a given similarity metric to take the (M_size/2) most dissimilar matching names and the (M_size/2) most similar non-matching names.

The training and test datasets are then collected in the two dictionary **TRAIN**={train_name: S^{name}_{train}, ...} and **TEST**={test_name: S^{name}_{test}, ...}

## How to run the experiments
The procedure to run the experiments is the following:
1. in config.py set:
  - PATH_EXP, with the path to the main folder of the project, containing the folder ''data/'' with the input
  - FOLDER_NAME, with the name of the folder where the output will be saved
  - X_COLS and Y_COL, with the name of the columns for the two features and the label
  - DATA_OPT[''source''], with the name of the input file contained in ''data/''. 
  - n_train and n_test, with the names and the sizes for the training and test sets. The sum between the sizes for the larger training and the larger test must be smaller then the number of all samples.
  - BATCH_LIST or BATCH_FIX, in such a way that it spanns all unlabeled samples
2. run Pipeline_experiments.py

This is the minimal actions required to run the experiments with a provided input. However, a synthetic dataset can be used by simply setting DATA_OPT[''source'']=None

**Experiments output are under ''DATE_FOLDER_NAME/'':**
- in ''DATE_FOLDER_NAME/table/'', the table with the models performance in standard supervised classification is saved with name ```table_performance```
- in ''DATE_FOLDER_NAME/active/'', four plots of the models performance in the active learning settings are saved.

## Workflow
The project workflow is implemented in **```experiments_pipeline(path, encoder_opt, siamese_options, X_cols, y_col, train, test)```**, and it consists of two different parts:
1. standard supervised classification
2. Active Learning

### Models performance in standard supervised classification
Three different types of classifiers are considered:  
- *baseline models*, i.e. Decision Stups using a string similarity metric as single feature
- *Random Forest*, using as features a number of string similarity metrics plus the number of words and the number of characters per string. 
- *Siamese neural network*, a Neural Network classifier on top of a learned LSTM embedding space of strings (for more information refer to our paper)

The performances of the models change as they are trained on training sets of different sizes. Each of the training sets is used to train each of the following 7 models:
- 5 baseline models, one for each of the following metrics: Levenstein distance (Lev), InDel ratio (ID), token set ratio (TSR), Jaccard similarity (Jac), and JW similarity (JW)
- a Random Forest with Lev, ID, TSR, Jac, JW, and number of words and the number of characters per string as features and hyperparameters ```max_depth=3```, ```n_estimators=100```, and ```class_weight=‘balanced’```
- a Siamese neural network, with parameters specified in ENCODER_OPT and SIAMESE_OPT

The method **```get_table(train, test, encoder_opt, siamese_opt, X_cols, y_col)```** generates and train the 7 models on the training sets specified in **train**. Later the method tests the models to compute precision, recall, f1 score, balanced accuracy, macro-averaged f1, and Matthew's correlation coefficient per each dataset in **test**. 
The output is the table containing the model performances in each test sets, which is later stored in ''DATE_FOLDER_NAME/table/'' folder.

### Models performance in the active learning settings
Active learning procedures are utilised to lower the costs of labelling while maximising the classifier performances. By starting with a limited number of labelled samples $X_l$ and a set of not-labelled-yet instances $X_u$, active learning considers labelling just a limited number of samples $X_c \subset X_u$ according to some query strategies in a multi-step procedure. Here we adopt an uncertainty sampling strategy where, at each step, instances of $X_u$ on which the prediction of the most updated model is less certain are selected. We denote this strategy as the **least-confident learner**.
These data points are then removed from $X_u$, labelled and then added to $X_l$ in order to train an (hopefully) improved version of the classifier. 
![ffig](https://user-images.githubusercontent.com/125964392/220958386-4743306a-ea7a-47ae-bca5-6d1b0690d7b1.jpg)

Both the Random Forest and the Siamese Neural Network are trained using the Active Learning approach. We utilise two types of learners: one is the least-confident learner described above, and the other is a **random learner**, in which the next-to-be labelled instances are picked at random. 
We assume that the initial labelled instances are the ones in the smaller training set $X^0_l=S^S_{train}$, while the total unlabeled istances are the instances in the difference between the larger and the smaller training sets, i.e. $X^0_u=S^L_{train}$\ $S^S_{train}$.
At the end of each iteration $j$, the classifier's performance are evaluated as follows: 
1. *pre-train batch test*: test the model $\mathcal{C}^{j}$ on the next-to-be-labelled instances, i.e. $X^{j}_{c}$
2. The model is trained with respect to the new training set $(X^{j}_l , y^{j}_l)$, thus obtaining $\mathcal{C}^{j+1}$
3. We train with respect to the new training set $(X^{j}_l , y^{j}_l)$, thus obtaining $\mathcal{C}^{j+1}$.
4. $\mathcal{C}^{j+1}$ is tested on:
  - the batch samples $X^{j}_{c}$, and we refer to it as the *post-train batch test*
  - the updated unlabelled set $X^{j}_u$, i.e. all the remaining unlabelled instances, and we refer to it as the *not-labelled-yet test*
  - the actual *test sets*

