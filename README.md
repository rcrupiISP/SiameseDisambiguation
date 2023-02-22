# SiameseDisambiguation
Python project for a disambiguation technique based on a Siamese Neural Network and a procedure of Active Learning.

> ### Authors & contributors:
> Alessandro Basile, Riccardo Crupi, Michele Grasso, Alessandro Mercanti, Daniele Regoli, Simone Scarsi, Shuyi Yang, Andrea Claudio Cosentini

To know more about this research work, please refer to our papers:

- Disambiguation of Company names via Deep Recurrent Networks


## Installation
Create a new environment based on Python 3.8.10 and install the requirements.

Python 3.8.10:
```
pip install -r requirements.txt
```
## Settings
The following is a brief description of the parameters collected in the file prj/app/config/config.py that can be modified by the user:
- Folder tree structure
  - PATH_EXP: is the path to the main folder containing the input (in ''data/'') and the output (in DATE_FOLDER_NAME). If not setted, a new folder ''experiments'' is created in the project directory.
  - FOLDER_NAME: is the name of the folder where the output are saved.
- Data
  - X_COLS: column names for the features (pair of strings)
  - Y_COL: column name for the label
  - DATA_OPT: dictionary with the settings for data loading and production. If DATA_OPT[''source''] a file name (with its extension), it looks for that file in ''data/''
  - n_train: dictionary with the name and the size of the training sets, in the form {name: size}
  - METHOD_ORDER: the method of a similarity metric -- one between 'jaro' (Jaro-Winkler), 'fuzz_ratio' (InDel ratio), 'fuzz_token_ratio' (token set ratio), 'jaccard', 'distance' (Levensthein).
  - n_test: dictionary with the name and the size of test sets in the form {name: [R_size, M_size]}, where R_size is the size of a randomly sampled test set (R_) and M_size is the size of a test sets, subsampled from R_ by using the method specified in METHOD_ORDER to select the most challenging cases.
- Neural Network settings
  - ENCODER_OPT: labelencoder settings specifying the maximum lenght of each string and the padding character used
  - SIAMESE_OPT: settings for the architecture, compiling and fitting of the siamese neural network
- Active learning
  - BATCH_LIST: list of batch sizes to use at each step during the active learning
  - BATCH_FIX: a fixed size for the batch, used if BATCH_LIST=None

## Input
The input data consist of a pair of strings, being names of real-world entities, and a binary target variable: we use the label 1 for the *matching names* (i.e. when the pair of names corresponds to the same entity), and label 0 for the *non-matching names* (i.e. when the pair of names refers to different entities).
### Get the data
The user can provide the data in two different ways:
- directly as a pandas DataFrame with 3 columns, two columns for the pair of names (X) and one for the lable (y), and stored in ''PATH_EXP/data/''. The name of the file and its extension has to be setted in DATA_OPT as DATA_OPT[''source'']=''file_name.file_ext''.
- as a pandas DataFrame with only the 2 columns for the pair of names (X) and **containing only matching names**. In this case, the function get_dataset() will generate the non-matching names by random pairing the names in the two columns X_COLS. The proportion between matching and non-matching names samples can be setted by tuning the DATA_OPT[''n_neg''] parameter in config.py, e.g. DATA_OPT[''n_neg'']=4 will produce 4 non-matching for each matching names. As in the previous case, the two columns dataframe has to be stored in ''PATH_EXP/data/'' and its name passed to DATA_OPT[''source''].

If DATA_OPT[''source'']=None or the file of the dataset can not be found in ''data/'' folder, a dummy dataset of acronyms with two columns is downoaded from ''https://raw.githubusercontent.com/krishnakt031990/Crawl-Wiki-For-Acronyms/master/AcronymsFile.csv'' and processed as described above.

### Training and test sets
The samples are then splitted in two groups for training and for testing, taking care of preserving the matching/non-matching ratio. 
By following the description in our paper, we generate a number of training datasets with differen sizes specified in ''n_train'' variable. These are obtained by hierarchical subsampling elements from the larger sample, i.e. from the training samples, select N_1 samples to form train_1 dataset, from train_1 dataset select N_2 samples (with N_1 > N_2) to form train_2 dataset, etc. 
The size of the test sets are specified in the ''n_test'' variable in the form {'test_name': [R_size, M_size]}. Two types of test sets are generated for each ''test_name'': the ''R_'' obtained by randomly selecting R_size elements from the test samples, and the ''M_'' obtained by ordering the ''R_'' samples by a given similarity metric to take the (M_size/2) most dissimilar matching names and the (M_size/2) most similar non-matching names.

## Workflow
The project workflow consists of the following steps:
