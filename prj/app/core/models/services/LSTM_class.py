import pandas as pd
import numpy as np
from numpy.random import seed
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Embedding, BatchNormalization, Lambda
from prj.app.config.config import PATH, X_COLS, Y_COL
from prj.app.core.data.services.load_dataset_experiments import split_dtf
seed(16)
tf.random.set_seed(7)


class Siamese:
    def __init__(self):
        self.encoder = None
        self.output_dim = None
        self.siamese_model = None
        self.input_shape = None
        self.structure = None

    @staticmethod
    def padding_and_labelencoding(le, string, tot_len, fill_char):
        """
        Apply the labelencoding to the string
        :param le: labelencoding map in the form of a dictionary {char: number}
        :param string: string to labelencoder
        :param tot_len: length of padding
        :param fill_char: padding char
        :return: a labelencoded string in the form of a list
        """
        padd_string = string.ljust(tot_len, fill_char)
        # Set to zero the unknown characters
        return [le[ch] if ch in le.keys() else 0 for ch in list(padd_string)]

    @staticmethod
    def fit_labelencoding(le, list_string, tot_len=500, fill_char=u"\u2797"):
        """
        Fit function for the labelencoder
        :param le: labelencoder to fit
        :param list_string: list of strings used for fitting
        :param tot_len: length of padding
        :param fill_char: padding char
        :return: a labelencoding map in the form of a dictionary {char: number} whith {fill_char: 0}
        """
        padd_list = [string.ljust(tot_len, fill_char) for string in list_string]
        le.fit([ch for words in padd_list for ch in list(words)])
        # Set padding character as 0
        map_le = {cl: (pl + 1) for pl, cl in enumerate(le.classes_)}
        map_le[fill_char] = 0
        return map_le

    @staticmethod
    def cosine_distance(vectorA, vectorB):
        product = tf.reduce_sum(vectorA * vectorB, axis=1, keepdims=True)
        normA = tf.reduce_sum(vectorA * vectorA, axis=1, keepdims=True)
        normB = tf.reduce_sum(vectorB * vectorB, axis=1, keepdims=True)
        return product / (normA * normB)

    @staticmethod
    def l_inf_distance(vectorA, vectorB):
        product = tf.reduce_max(tf.math.abs(vectorA - vectorB), axis=1, keepdims=True)
        return product

    @staticmethod
    def l1_distance(vectorA, vectorB):
        product = tf.reduce_sum(tf.math.abs(vectorA - vectorB), axis=1, keepdims=True)
        return product

    @staticmethod
    def l2_distance(vectorA, vectorB):
        product = tf.reduce_sum((vectorA - vectorB) ** 2, axis=1, keepdims=True)
        return product

    @staticmethod
    def abs_diff(vectorA, vectorB):
        product = tf.math.abs(vectorA - vectorB)
        return product

    @staticmethod
    def rebalancing(dtf_in):
        """
        Function used to rebalance a dataframe. The class with less samples is the minority class
        :param dtf_in: dataframe to rebalance
        :return: a dataframe with same namber of sample for the two classes. The minoritary class is rebalanced by adding
        samples randomly chosen within the same class
        """
        df_pos = dtf_in[dtf_in['y'] == 1]
        df_neg = dtf_in[dtf_in['y'] == 0]
        if df_pos.shape[0] == df_neg.shape[0]:
            return dtf_in
        else:
            if df_pos.shape[0] < df_neg.shape[0]:
                df_over = df_neg.copy()
                df_under = df_pos.copy()
            else:
                df_over = df_pos.copy()
                df_under = df_neg.copy()
            prop = (df_over.shape[0] / df_under.shape[0]) - 1
            dtf_out = pd.concat([dtf_in, df_under.sample(frac=prop, random_state=11, replace=True)], ignore_index=True)
            dtf_out = dtf_out.sample(frac=1, random_state=0).reset_index(drop=True)
            return dtf_out

    def get_map_encoding(self, opt):
        """
        Generate the labelencoder
        :param opt: settings
        :return: the labelencoder map
        """
        label_encoder = preprocessing.LabelEncoder()

        print("Generate labelencoding map")
        dtf_to_encode = opt['encode_on']
        train_labelling = []
        [train_labelling.extend(dtf_to_encode[c].to_list()) for c in dtf_to_encode.columns]
        map_le = self.fit_labelencoding(label_encoder, train_labelling, tot_len=opt['padding'])
        opt['map'] = map_le

        self.encoder = opt
        self.output_dim = len(map_le)
        self.input_shape = opt['padding']

    def data_preprocessing(self, df_data):
        """
        Preprocessing of the data before being used in the Neural Network.
        It is done in 3 steps:
                                - string cleaning: remove multiple whitespaces and put all chars in uppercase
                                - labelencoding
                                - use the tokenised character-wise string as new feature
        :param df_data: dataframe to preprocess
        :return: columns names for the tokenised character-wise strings A and B, the label column name, and a dataframe
        having 2*opt['padding'] columns for features and 1 column (y_col) for label
        """
        encoder = self.encoder
        X_cols = X_COLS
        y_col = Y_COL
        data_pre = df_data.copy()

        # Cleaning string: uppercase and remove multiple whitespaces
        for c in X_cols:
            data_pre[c] = data_pre[c].str.upper()
            data_pre[c] = data_pre[c].apply(lambda x: " ".join(x.split()))

        # Apply padding and labelencoding
        data_pre['name_A'] = data_pre[X_cols[0]].apply(lambda row:
                                                          self.padding_and_labelencoding(encoder['map'],
                                                                                         row,
                                                                                         encoder['padding'],
                                                                                         encoder['char']))
        data_pre['name_B'] = data_pre[X_cols[1]].apply(lambda row:
                                                          self.padding_and_labelencoding(encoder['map'],
                                                                                         row,
                                                                                         encoder['padding'],
                                                                                         encoder['char']))
        x_colsA = [f'a_{i}' for i in range(encoder['padding'])]
        x_colsB = [f'b_{i}' for i in range(encoder['padding'])]

        data_pre_1 = pd.DataFrame(data_pre['name_A'].to_list(), columns=x_colsA)
        data_pre_2 = pd.DataFrame(data_pre['name_B'].to_list(), columns=x_colsB)
        data_shape = pd.concat([data_pre_1, data_pre_2], axis=1)
        data_shape[y_col] = data_pre[y_col]

        return x_colsA, x_colsB, y_col, data_shape

    def model_architecture(self,
                            architecture='distance',
                           learning_rate=0.0001,
                           b1=0.8,
                           b2=0.9,
                           drop_out=0.1,
                           ):
        """
        Structure of the neural network. It has two main parts: an embedding (applyied to the two labelencoded strings),
        and the layers (to concatenate the two embedded strings and apply the RNN)
        :param architecture: option used to select which functions are applyied to the two embeddings
        :param learning_rate: learning rate in the Nadam compiler
        :param b1: \beta_1 in the Nadam compiler
        :param b2: \beta_2 in the Nadam compiler
        :param drop_out: dropout level
        :return: the model structure
        """
        tf.random.set_seed(7)

        input_shape = self.input_shape
        output_dim = self.output_dim

        #Embedding
        inputs = tf.keras.Input(shape=input_shape, name='input_string')
        embedding_model = Embedding(input_dim=input_shape, output_dim=output_dim, mask_zero=True)(inputs)
        embedding_model = Dropout(drop_out)(embedding_model)
        embedding_model = LSTM(16, go_backwards=True)(embedding_model)
        embedding_model = Dropout(drop_out)(embedding_model)
        embedding_model_final = tf.keras.Model(inputs, embedding_model, name='embedding_model')
        embedding_model_final.summary()

        #Layers
        inputA = tf.keras.Input(shape=input_shape, name='input_string_A')
        inputB = tf.keras.Input(shape=input_shape, name='input_string_B')
        featA = tf.keras.Model(inputs, embedding_model, name='embedding_model_A')(inputA)
        featB = tf.keras.Model(inputs, embedding_model, name='embedding_model_B')(inputB)

        cos = Lambda(lambda x: self.cosine_distance(x[0], x[1]), name='cosine_distance')([featA, featB])
        l1 = Lambda(lambda x: self.l1_distance(x[0], x[1]), name='l1_distance')([featA, featB])
        l2 = Lambda(lambda x: self.l2_distance(x[0], x[1]), name='l2_distance')([featA, featB])
        l_inf = Lambda(lambda x: self.l_inf_distance(x[0], x[1]), name='l_inf_distance')([featA, featB])
        diff = Lambda(lambda x: self.abs_diff(x[0], x[1]), name='abs_difference')([featA, featB])

        if architecture == 'distance':
            feat = Concatenate()([cos, l1, l2, l_inf])
        elif architecture == 'difference':
            feat = self.abs_diff(featA, featB)
        else:
            print(f"{architecture}, good choice! I know you want dist_and_diff.")
            feat = Concatenate()([cos, l1, l2, l_inf, diff])

        feat = Dense(32, activation='relu')(feat)
        feat = BatchNormalization()(feat)
        feat = Dropout(drop_out)(feat)

        feat = Dense(16, activation='relu')(feat)
        feat = BatchNormalization()(feat)
        feat = Dropout(drop_out)(feat)

        output = Dense(1, activation='sigmoid')(feat)
        model = tf.keras.Model(inputs=[inputA, inputB], outputs=output)
        model.summary()

        tf.random.set_seed(16)
        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=learning_rate,
                                                          beta_1=b1,
                                                          beta_2=b2,
                                                          ),
                        loss='binary_crossentropy',
                        metrics=['accuracy', 'mae'])
        return model

    def fit(self, dtf_in, fit_options):
        """
        Fit function
        :param dtf_in: dataframe used tor training
        :param fit_options: setting options for compiler and fit
        :return:
        """

        # Apply rebalancing
        if fit_options['re-balance']:
            dtf = self.rebalancing(dtf_in)
        else:
            dtf = dtf_in.copy()

        x_colsA, x_colsB, y_col, df_preproc = self.data_preprocessing(dtf)
        train_df, val_df = split_dtf(df_preproc, split=fit_options['size'], level=y_col)
        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        if self.siamese_model is None:
            siamese_model = self.model_architecture(architecture=fit_options['architecture'],
                                                    learning_rate=fit_options['lr'],
                                                    b1=fit_options['beta_1'],
                                                    b2=fit_options['beta_2'],
                                                    )
            self.siamese_model = siamese_model
        else:
            siamese_model = self.siamese_model

        # set the callbacks
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         patience=fit_options['patience'],
                                                         restore_best_weights=True,
                                                         )
        filepath = f'{PATH}models/ckpt'
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                        monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        mode='max')
        callbacks_list = [earlystopping, checkpoint]

        # use an adaptive batch size witch depends on the size of the training dataset
        batch_rescaled = int(df_preproc.shape[0]/fit_options['n_batch'])
        tf.random.set_seed(7)
        np.random.seed(2022)

        history = siamese_model.fit(x=[train_df[x_colsA], train_df[x_colsB]],
                                    y=train_df[y_col],
                                    batch_size=batch_rescaled,
                                    epochs=fit_options['epochs'],
                                    validation_data=([val_df[x_colsA], val_df[x_colsB]], val_df[y_col]),
                                    shuffle=True,
                                    callbacks=callbacks_list,
                                    )
        siamese_model.load_weights(filepath)
        self.siamese_model = siamese_model

    def predict(self, dtf_in):
        """
        Model predictions over the dtf_in dataset
        :param dtf_in: data to classify
        :return: an array of class for each sample in dtf_in
        """
        dtf = dtf_in.copy()
        x_colsA, x_colsB, y_col, df_pr = self.data_preprocessing(dtf)
        pred = (self.siamese_model.predict([df_pr[x_colsA], df_pr[x_colsB]]) > 0.5).astype("int32")
        return pred[:, 0]

    def predict_proba(self, dtf_in):
        """
        Probability predictions over the dtf_in dataset
        :param dtf_in: data to classify
        :return: a multidimensional array where for each sample in dtf_in shows [prob_0, prob_1], i.e.
         the probability prediction associated to each class
        """
        dtf = dtf_in.copy()
        x_colsA, x_colsB, y_col, df_pr = self.data_preprocessing(dtf)
        y_pred = self.siamese_model.predict([df_pr[x_colsA], df_pr[x_colsB]])
        return np.array(list(zip(1 - y_pred[:, 0], y_pred[:, 0])))
