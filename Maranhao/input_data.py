import numpy as np, pandas as pd
import sys

def open_file(file_to_read, f):

    dataset = pd.read_csv(file_to_read, delimiter = " ", header=None, names=[file_to_read.replace('.txt', '')])
    global joined_data
    joined_data = pd.DataFrame(dataset)
    #join_data(data_to_join, f)
    
    return

def join_data(data, f):

    if f==0:
        global joined_data
        joined_data=data
    else:
        if joined_data.shape[0] != data.shape[0]:
            print ('        The sample size of input variables is different.\n        All the samples must have the same size.')
            sys.exit()
        joined_data = pd.concat([joined_data, data], axis=1)

    return

def drop_NA(data, return_only_values):

    data.dropna(inplace=True)
    data.astype(float)
    global dates
    global joined_data_values
    
    dates = (data.index).tolist()
    
    if return_only_values:
        joined_data_values = data.values
        return joined_data_values, dates
    else:
        joined_data_values = data
        return joined_data_values, dates
    
    return

def input_data_main(input_files):

    for f, file in enumerate(input_files):
        open_file(file, f)

    return joined_data