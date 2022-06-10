import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
import tensorflow as tf
import matplotlib.pyplot as plt
from math import sqrt
import input_data, ChangeDataset, hyperparameter_tunning, statistical_parameters
import sys, os, shutil

basin = 'Maranhao'
fin_var_out = 'flow.txt'
fin_var_in1 = 'precipitation.txt'
fin_var_in2 = 'temperature.txt'

sample_division = [0.7, 0.2, 0.1] #train, validation, test

accumulate_values = False #True or False
average_values = False #True or False
delay_values = True #True or False

accumulate_periods = [[fin_var_in1, 2, 3, 4, 5, 10]]
average_periods = []
delay_periods = [[fin_var_in1, 1, 2, 3, 4, 5, 6, 7], [fin_var_in2, 1, 2, 3, 4, 5, 6, 7]]


############################################################################################
#open input files and join to a unique dataset
files_to_join = [fin_var_out, fin_var_in1, fin_var_in2]
data = input_data.input_data_main(files_to_join)

#Accumulate or average data
if accumulate_values:
    data = ChangeDataset.accumulate(data, accumulate_periods)
    
if average_values:
    data = ChangeDataset.average(data, average_periods)
    
if delay_values:
    data = ChangeDataset.delay(data, delay_periods)

print(data.head())
data=input_data.drop_NA(data, True)

####Prepare data
#optimization()
#Divide data into forecasted property and forcing properties. Change the shape of numpy arrays nedeed to keras model operation
x_dataset = data[:,1:]
y_dataset = data[:,0]
y_dataset = np.reshape(y_dataset, (-1,1))

# change forcing properties scale
scaler_x = MinMaxScaler(feature_range=(0,0.9)) #feature_range=(0,0.9)
x_dataset_scale = scaler_x.fit_transform(x_dataset)

#change forecasted property scale
scaler_y = MinMaxScaler(feature_range=(0,0.9)) #feature_range=(0,0.9)
y_dataset_scale = scaler_y.fit_transform(y_dataset)

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

optim=['Nadam', 'Adamax', 'RMSprop', 'Adam', 'Adagrad', 'SGD'] #'Nadam', 'Adamax', 'RMSprop', 'Adam', 'Adagrad', 

current_dir = os.getcwd()

for op in optim:
    case_study=basin+'_'+op
    model, params=hyperparameter_tunning.tuner(case_study, x_train, y_train, x_validation, y_validation, op) # #ep, 

    #predictions
    predictions_scale = model.predict(x_test)
    predictions = scaler_y.inverse_transform(predictions_scale)

    def plot(tv, true, predicted):

        obs = np.concatenate((tv, true))
        tv = np.reshape(tv, (tv.shape[0],1))
        pred = np.concatenate((tv, predicted))
        
        plt.figure(figsize=(10,6))
        plt.plot(obs, label='Observed') #,linewidth=5)
        plt.plot(pred, label='Predictions',color='y')

        plt.legend()
        plt.savefig(current_dir+'/'+case_study+'/'+'best_total.png')
        
        plt.figure(figsize=(10,6))
        plt.plot(true, label='Observed') #,linewidth=5)
        plt.plot(predicted, label='Predictions',color='y')

        plt.legend()
        plt.savefig(current_dir+'/'+case_study+'/'+'best_test.png')

        # calculate statistical paramters
        nse = statistical_parameters.nse(np.concatenate((true, predicted), axis=1))
        r2 = statistical_parameters.r2(np.concatenate((true, predicted), axis=1))
        pbias = statistical_parameters.pbias(np.concatenate((true, predicted), axis=1))
        rmse = statistical_parameters.rmse(np.concatenate((true, predicted), axis=1))
        
        return nse, r2, pbias, rmse

    # Results
    y_train_validation=y_dataset[:train_size+val_size,:]
    y_test=y_dataset[train_size+val_size:,:]
    stats=plot(y_train_validation, y_test, predictions)
    
    fin = open(current_dir+'/'+case_study+'/'+'best_params.txt', 'w')
    fin.write('Train, validation, test sizes: '+str(y_train.shape)+str(y_validation.shape)+str(y_test.shape)+'\n')
    fin.write(str(params)+'\n')
    fin.writelines('NSE: '+str(stats[0])+'\n'+'R2: '+str(stats[1])+'\n'+'PBIAS: '+str(stats[2])+'\n'+'RMSE: '+str(stats[3])+'\n')
    fin.close()
