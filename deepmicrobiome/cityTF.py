import pandas as pd
import numpy as np

#working, modify for state and city
full = pd.read_csv('fulltable.csv', delimiter='\t')
full = full.drop('ID', axis=1)
full = full.drop('Country', axis=1)
full = full.drop('State', axis=1)
full = full.drop('BodySite', axis=1)
full_X = full.drop('City', axis=1)
X = full_X.values
full_y = full['City']
y = full_y.values

#no scaler is needed for X data because it is already normalized
#TTS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

#One Hot Encoding the categorical y data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder() #set up le
integer_encoded_all = label_encoder.fit_transform(y) #ONLY FOR CITY
integer_encoded_train = label_encoder.transform(y_train) #puts into integers up to 139 for train
integer_encoded_test = label_encoder.transform(y_test) #^... for test

onehot_encoder = OneHotEncoder(sparse=False) #set up 1hc

integer_encoded_all = integer_encoded_all.reshape(len(integer_encoded_all), 1)
integer_encoded_train = integer_encoded_train.reshape(len(integer_encoded_train), 1)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)

#use y1 in the training and testing
y1 = onehot_encoder.fit_transform(integer_encoded_all)
y1_train = onehot_encoder.transform(integer_encoded_train)
y1_test = onehot_encoder.transform(integer_encoded_test)

#y1.shape

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

#setting up possible hyperparameters with multiple layers
HP_NUM_UNITS_1 = hp.HParam('num_units_1', hp.Discrete([1500, 2000, 2500]))
HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([1500, 2000, 2500]))
HP_NUM_UNITS_3 = hp.HParam('num_units_3', hp.Discrete([1500, 2000, 2500]))
HP_NUM_UNITS_4 = hp.HParam('num_units_4', hp.Discrete([1500, 2000, 2500]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2, 0.25))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['nadam', 'adam']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS_1, HP_NUM_UNITS_2, HP_NUM_UNITS_3, HP_NUM_UNITS_4, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
)

from datetime import datetime
logdir = 'logs\\hparam_tuning' + '\\' + datetime.now().strftime("%Y-%m-%d")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=60) #valloss needs a val set, whihc i dont have rn, so replace with loss
board = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', profile_batch=2, embeddings_freq=1)


def train_test_model(hparams):
    model = Sequential()
    model.add(Dense(units=993, activation='relu'))
    model.add(Dense(hparams[HP_NUM_UNITS_1], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(hparams[HP_NUM_UNITS_2], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(hparams[HP_NUM_UNITS_3], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(hparams[HP_NUM_UNITS_4], activation='relu'))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(units=139, activation='softmax')) #change based on data c/s/c

    model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )
    
    model.fit(X_train, y1_train, epochs=1000, validation_data=(X_test, y1_test), callbacks=[early_stop, board, hp.KerasCallback(logdir, hparams)])#commenting out for now:, callbacks=[tf.keras.callbacks.TensorBoard(logdir), hp.KerasCallback(logdir, hparams)])
    # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(X_test, y1_test)
    return accuracy, model

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy, model = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
    return accuracy, model

#declaring the things that ill look in to extract hte best performing model
highest_acc = 0
highest_sess = 0
models = []
session_num = 0

#refactored so that the best model is outputted
for num_units_1 in HP_NUM_UNITS_1.domain.values:
    for num_units_2 in HP_NUM_UNITS_2.domain.values:
        for num_units_3 in HP_NUM_UNITS_3.domain.values:
            for num_units_4 in HP_NUM_UNITS_4.domain.values:
                for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    for optimizer in HP_OPTIMIZER.domain.values:
                        hparams = {
                            HP_NUM_UNITS_1: num_units_1,
                            HP_NUM_UNITS_2: num_units_2,
                            HP_NUM_UNITS_3: num_units_3,
                            HP_NUM_UNITS_4: num_units_4,
                            HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        current_acc, current_mod = run('logs/hparam_tuning/' + run_name, hparams)
                        models.append(current_mod)
                        if (highest_acc < current_acc):
                            highest_acc = current_acc
                            highest_sess = session_num
                            #print('highest!')
                        session_num += 1

best_model = models[highest_sess]

#classification analysis with predictions
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#both no longer one hot encoded but label encoded
pred = np.argmax(best_model.predict(X_test), axis=-1)
check = np.argmax(y1_test, axis=-1) 

#evaluation
report = classification_report(check, pred)
matrix = confusion_matrix(check, pred)
score = accuracy_score(check, pred)

#all f or deploy
integer_encoded_all = label_encoder.fit_transform(y)
integer_encoded_all = integer_encoded_all.reshape(len(integer_encoded_all), 1)
y1 = onehot_encoder.fit_transform(integer_encoded_all)

#train and predict on all
best_model.fit(X, y1, callbacks=[early_stop], epochs=1000)
eval = best_model.evaluate(X, y1)
final_loss = eval[0]
final_acc = eval[1]

#saving it to the directory
best_model.save('citymodel.h5')
import joblib
joblib.dump(label_encoder, 'citylabel.pkl')
joblib.dump(onehot_encoder, 'cityonehot.pkl')

#output file
f = open('cityout.txt', 'w')
f.write('score:\n')
f.write(str(score))
f.write('matrix:\n')
f.write(str(matrix))
f.write('report:\n')
f.write(str(report))
f.write('final:\n')
f.write('loss: ' + str(final_loss) + ' acc: ' + str(final_acc))
f.close()