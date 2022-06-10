import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
from tensorflow.keras.models import Sequential, load_model, save_model
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
from keras.losses import mean_squared_error
from keras.activations import relu, elu, sigmoid, softmax, softplus, softsign, tanh, selu, exponential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from math import sqrt
import input_data, ChangeDataset, statistical_parameters
import sys, os, shutil

n_model = r'F:\Ana_Projetos\8_Omega\_ANN\20012008\RerunOriginalDataset\Model_94'
#fin_var_out = 'flow.txt'
fin_var_in1 = 'precipitation.txt'
fin_var_in2 = ''#'temperature.txt'

accumulate_values = True #True or False
average_values = False #True or False
delay_values = True #True or False

accumulate_periods = [[fin_var_in1, 10, 30, 60]]
average_periods = []
delay_periods = [[fin_var_in1, 1, 2, 3, 4, 5, 6, 7]]

############################################################################################
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
files_to_join = [fin_var_in1]
data = input_data.input_data_main(files_to_join)

#Accumulate or average data
if accumulate_values:
    data = ChangeDataset.accumulate(data, accumulate_periods)
    
if average_values:
    data = ChangeDataset.average(data, average_periods)
    
if delay_values:
    data = ChangeDataset.delay(data, delay_periods)

print(data.head)
data, dates=input_data.drop_NA(data, True)

####Prepare data
# change forcing properties scale
with open("scaler_x.pkl", "rb") as infile_x:
    scaler_x = pkl.load(infile_x)
    x_dataset_scale = scaler_x.transform(data)
x_dataset_scale = x_dataset_scale.reshape(x_dataset_scale.shape[0], x_dataset_scale.shape[1], 1)

# load model
m = load_model(n_model)

#predictions
predictions_scale = m.predict(x_dataset_scale)

with open("scaler_y.pkl", "rb") as infile_y:
    scaler_y = pkl.load(infile_y)
    predictions = scaler_y.inverse_transform(predictions_scale)

if len(dates) != predictions.shape[0]:
    print ("ERROR: Number of dates different from number of predictions!")
    sys.exit()

with open('PredictedFlow.txt', 'w') as filehandle:
    for lin in range(len(dates)):
        filehandle.write(str(dates[lin])+' '+str(predictions[lin][0])+'\n')