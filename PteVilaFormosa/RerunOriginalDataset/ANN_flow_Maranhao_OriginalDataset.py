import numpy as np
#from numpy.random import seed
#import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model
from keras.optimizers import Nadam
from keras.losses import mean_squared_error
from keras.activations import relu, elu
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import pickle as pkl
import matplotlib.pyplot as plt
from math import sqrt
import input_data, ChangeDataset, statistical_parameters
import sys, os, shutil

#hashseed = os.getenv('PYTHONHASHSEED')
#if not hashseed:
#    os.environ['PYTHONHASHSEED'] = '0'
#    os.execv(sys.executable, [sys.executable] + sys.argv)

fin_var_out = 'flow.txt'
fin_var_in1 = 'precipitation.txt'
fin_var_in2 = ''#'temperature.txt'

#number_of_threads_to_use = 3

sample_division = [0.7, 0.2, 0.1] #train, validation, test

accumulate_values = True #True or False
average_values = False #True or False
delay_values = True #True or False

accumulate_periods = [[fin_var_in1, 10, 30, 60]]
average_periods = []
delay_periods = [[fin_var_in1, 1, 2, 3, 4, 5, 6, 7]]

############################################################################################
def model_fit(x_t, x_v, y_t, y_v, m):

    # Nadam
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=10, input_shape=(x_train.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=8, kernel_size=16, padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    model.add(Dense(units=10,activation='relu'))

    # then we finish again with completely standard Keras way
    model.add(Dense(1, activation='elu'))

    model.compile(optimizer=Nadam(lr=0.001, epsilon=1e-8), loss='mean_squared_error', metrics=['mean_squared_error'])

    model.fit(x_t,y_t, epochs=300, batch_size=10, validation_data=(x_v, y_v), verbose=0)
    
    filepath = 'Model_'+str(run)
    model.save(filepath)

    return model
    
def plot(true, predicted, graph_name):

    plt.figure(figsize=(10,6))
    plt.plot(true, label='Observed')
    plt.plot(predicted, label='Predictions',color='y')

    plt.legend()
    plt.savefig(graph_name+'.png')
    plt.close()

    # calculate statistical paramters
    nse = statistical_parameters.nse(np.concatenate((true, predicted), axis=1))
    r2 = statistical_parameters.r2(np.concatenate((true, predicted), axis=1))
    pbias = statistical_parameters.pbias(np.concatenate((true, predicted), axis=1))
    rmse = statistical_parameters.rmse(np.concatenate((true, predicted), axis=1))
    
    return nse, r2, pbias, rmse

#open input files and join to a unique dataset
files_to_join = [fin_var_out, fin_var_in1]
data = input_data.input_data_main(files_to_join)

#Accumulate or average data
if accumulate_values:
    data = ChangeDataset.accumulate(data, accumulate_periods)
    
if average_values:
    data = ChangeDataset.average(data, average_periods)
    
if delay_values:
    data = ChangeDataset.delay(data, delay_periods)

print(data.head)
data=input_data.drop_NA(data, True)

####Prepare data
#optimization()
#Divide data into forecasted property and forcing properties. Change the shape of numpy arrays nedeed to keras model operation
x_dataset = data[:,1:]
y_dataset = data[:,0]
y_dataset = np.reshape(y_dataset, (-1,1))

# change forcing properties scale
scaler_x = MinMaxScaler(feature_range=(0,0.9))
x_dataset_scale = scaler_x.fit_transform(x_dataset)
with open("scaler_x.pkl", "wb") as outfile_x:
    pkl.dump(scaler_x, outfile_x)

#change forecasted property scale
scaler_y = MinMaxScaler(feature_range=(0,0.9))
y_dataset_scale = scaler_y.fit_transform(y_dataset)
with open("scaler_y.pkl", "wb") as outfile_y:
    pkl.dump(scaler_y, outfile_y)

sys.exit()
#get number of lines in each dataset (train, validation and test)
n_rows = data.shape[0]
train_size = int(n_rows * sample_division[0])
val_size = int(n_rows * sample_division[1])
test_size = int(n_rows * sample_division[2])

#divide forcing properties into train, validation and test datasets
x_train = x_dataset_scale[:train_size, :]
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

x_validation = x_dataset_scale[train_size:train_size+val_size, :]
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], 1)

x_test = x_dataset_scale[train_size+val_size:, :]
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#divide forecasted property into train, validation and test datasets and reshape
y_train = y_dataset_scale[:train_size, :]
y_train = np.reshape(y_train, (-1,1))
y_validation = y_dataset_scale[train_size:train_size+val_size, :]
y_validation = np.reshape(y_validation, (-1,1))
y_test = y_dataset_scale[train_size+val_size:, :]
y_test = np.reshape(y_test, (-1,1))

number_of_runs = 100
for run in range(number_of_runs):
    #seed(run)
    #tensorflow.random.set_seed(run)

    print("Working on run " + str(run) + ".")
    m = model_fit(x_train, x_validation, y_train, y_validation, run)

    #predictions
    predictions_scale = m.predict(x_test)
    predictions = scaler_y.inverse_transform(predictions_scale)

    #observed values
    observed_y_values = scaler_y.inverse_transform(y_test)

    # Results
    stats=plot(observed_y_values, predictions, 'Model_'+str(run))

    with open('Model_'+str(run)+'.txt', 'w') as filehandle:
        filehandle.write(", ".join(map(str, stats)))
        filehandle.write("\n".join(map(str, predictions)))
        filehandle.write("\n".join(map(str, observed_y_values)))