import numpy as np
import pandas as pd
import sys

def accumulate(data_set, acc_periods):

    for set_acc in acc_periods:
        dataset = pd.read_csv(set_acc[0], delimiter = " ", names=[set_acc[0].replace('.txt', '')])
        data = pd.DataFrame(dataset)
        column = data.columns[0]
        for period in set_acc[1:]:
            if 'pcp' in column or 'precipitation' in column or 'prec' in column or \
            'in' in column or 'inflow' in column:
                data_sum = data.rolling(min_periods=period, window=period, closed='right').sum()
            else:
                data_sum = data.rolling(min_periods=period, window=period).sum().shift()
            
            data_sum.columns=[column+'_'+str(period)+'days']
            data_set = pd.concat([data_set, data_sum], axis=1)

    return data_set
    
    
def average(data_set, ave_periods):

    for set_ave in ave_periods:
        dataset = pd.read_csv(set_ave[0], delimiter = " ", names=[set_ave[0].replace('.txt', '')])
        data = pd.DataFrame(dataset)
        column = data.columns[0]
        for period in set_ave[1:]:
            if 'pcp' in column or 'precipitation' in column or 'prec' in column or \
            'in' in column or 'inflow' in column:
                data_mean = data.rolling(min_periods=period, window=period, closed='right').mean()
            else:
                data_mean = data.rolling(min_periods=period, window=period).mean().shift()
            
            data_mean.columns=[column+'_'+str(period)+'days']
            data_set = pd.concat([data_set, data_mean], axis=1)

    return data_set
    
def delay(data_set, delay_periods):

    for set_del in delay_periods:
        dataset = pd.read_csv(set_del[0], delimiter = " ", names=[set_del[0].replace('.txt', '')])
        data = pd.DataFrame(dataset)
        column = data.columns[0]
        for period in set_del[1:]:
            data_shift = data.shift(period, axis=0)
            
            data_shift.columns=[column+'_'+str(period)+'days']
            data_set = pd.concat([data_set, data_shift], axis=1)

    return data_set