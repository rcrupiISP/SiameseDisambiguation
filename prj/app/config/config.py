import os
from datetime import datetime
import pathlib

# Folder settings
now = datetime.now()
DATE = now.strftime("%Y-%m-%d")

PATH_EXP = None
if PATH_EXP is None:
    local_path_prj = pathlib.Path(__file__).parent.parent.parent.parent.resolve()
    PATH_EXP = f'{local_path_prj}/experiments/'
    if not os.path.exists(PATH_EXP):
        os.makedirs(PATH_EXP)

if not os.path.exists(PATH_EXP+'data/'):
    os.makedirs(PATH_EXP+'data/')

FOLDER_NAME = f'git'
PATH = PATH_EXP + f"{DATE}_{FOLDER_NAME}/"

# Training and test sets
X_COLS = ['name_1', 'name_2']
Y_COL = 'y'

SOURCE_DEFAULT = 'https://raw.githubusercontent.com/krishnakt031990/Crawl-Wiki-For-Acronyms/master/AcronymsFile.csv'

DATA_OPT = {'frac': 0.5,  # if generate from positive samp., shuffle frac% of the dtf
             'n_neg': 4,   # if generate from positive samp., append n_neg non-matchings per matching couple
             'source': None,
            }

n_train = {
    'large': 6000,  # Number of samples for the large training dataset
    'medium': 2000,  # Number of samples for the medium training dataset
    'small': 100,   # Number of samples for the small training dataset
}

METHOD_ORDER = 'jaro'   # Method used to order and sample the test set
n_test = {
    'O': [3000, 200]    # Number of samples for random sampled (R_) and method ordered training sets: [R_, method]
}

# Neural network settings
# # Encoder settings
ENCODER_OPT = {'padding': 300,
               'char': u"\u2797",
               }

# # Siamese architecture, compiling, and fitting settings
SIAMESE_OPT = {'size': 0.25,  # portion for validation, and leaves 1-size for train
               'lr': 0.0002,  # learning rate
               'beta_1': 0.8,   # Nadam beta_1 parameter
               'beta_2': 0.9,   # Nadam beta_2 parameter
               'patience': 20,  # patience in earlystopping callbacks
               'epochs': 90,    # epochs used in fit
               'n_batch': 32,  # number of batchs per dataset: batch_size = dtf.shape[0]/n_batch
               'architecture': 'dist_and_diff',  # difference, distance or dist_and_diff (default)
               're-balance': True,      # if True multiply 4 times the positives
               }

# Active learning batch settings
BATCH_LIST = [100, 200, 400, 800, 800, 800, 1400, 1400]
#[100, 200, 400, 800, 800, 800, 1400, 1400] 100, 200, 400, 800, 1600, 2400, 3200, 4600, 6000
BATCH_FIX = 0