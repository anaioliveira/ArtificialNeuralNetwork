from tensorflow.keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
from keras.losses import mean_squared_error, mean_absolute_percentage_error
from keras.activations import relu, elu, sigmoid, softmax, softplus, softsign, tanh, selu, exponential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, LSTM
import tensorflow as tf
import kerastuner as kt

class MyTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
    
        # batch size and epochs
        kwargs['batch_size'] = hp.Choice('batch_size', values=[1, 10, 50, 100])
        #kwargs['batch_size'] = hp.Int('batch_size', 10, 50, step=10)
        kwargs['epochs'] = hp.Int('epochs', 50, 250, step=50, default=150)

        super(MyTuner, self).run_trial(trial, *args, **kwargs)

def build_model_Adam(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    nodes_l1=hp.Choice('input_nodes', values=[8, 16, 32, 64])
    model.add(LSTM(units=nodes_l1, input_shape=(None, x_train_aux.shape[2]), return_sequences=True))

    # if we want to also test for number of layers and shapes, that's possible
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    if n_hidden_layers != 0:
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1*3, return_sequences=True))
   
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1, return_sequences=True))
   
    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=Adam(lr=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_SGD(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    nodes_l1=hp.Choice('input_nodes', values=[8, 16, 32, 64])
    model.add(LSTM(units=nodes_l1, input_shape=(None, x_train_aux.shape[2]), return_sequences=True))

    # if we want to also test for number of layers and shapes, that's possible
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    if n_hidden_layers != 0:
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1*3, return_sequences=True))
   
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1, return_sequences=True))
   
    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=SGD(lr=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_Nadam(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    nodes_l1=hp.Choice('input_nodes', values=[8, 16, 32, 64])
    model.add(LSTM(units=nodes_l1, input_shape=(None, x_train_aux.shape[2]), return_sequences=True))

    # if we want to also test for number of layers and shapes, that's possible
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    if n_hidden_layers != 0:
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1*3, return_sequences=True))
   
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1, return_sequences=True))
   
    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=Nadam(lr=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_Adagrad(hp):
    
    # next we can build the model exactly like we would normally do it
    model = Sequential()
    nodes_l1=hp.Choice('input_nodes', values=[8, 16, 32, 64])
    model.add(LSTM(units=nodes_l1, input_shape=(None, x_train_aux.shape[2]), return_sequences=True))

    # if we want to also test for number of layers and shapes, that's possible
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    if n_hidden_layers != 0:
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1*3, return_sequences=True))
   
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1, return_sequences=True))
   
    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=Adagrad(lr=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_RMSprop(hp):
    
    # next we can build the model exactly like we would normally do it
    model = Sequential()
    nodes_l1=hp.Choice('input_nodes', values=[8, 16, 32, 64])
    model.add(LSTM(units=nodes_l1, input_shape=(None, x_train_aux.shape[2]), return_sequences=True))

    # if we want to also test for number of layers and shapes, that's possible
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    if n_hidden_layers != 0:
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1*3, return_sequences=True))
   
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1, return_sequences=True))
   
    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=RMSprop(lr=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_Adamax(hp):
    
    # next we can build the model exactly like we would normally do it
    model = Sequential()
    nodes_l1=hp.Choice('input_nodes', values=[8, 16, 32, 64])
    model.add(LSTM(units=nodes_l1, input_shape=(None, x_train_aux.shape[2]), return_sequences=True))

    # if we want to also test for number of layers and shapes, that's possible
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    if n_hidden_layers != 0:
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1*3, return_sequences=True))
   
        if n_hidden_layers == 1:
            model.add(LSTM(units=nodes_l1, return_sequences=True))
            
        if n_hidden_layers == 2:
            model.add(LSTM(units=nodes_l1*2, return_sequences=True))
            model.add(LSTM(units=nodes_l1, return_sequences=True))
   
    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=Adamax(lr=hp_learning_rate), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def tuner(case_study, x_train, y_train, x_validation, y_validation, opt):

    tf.keras.backend.clear_session()
    
    global x_train_aux
    x_train_aux = x_train
    n_trials=1000
    
    if opt == 'Adam':
        # run model with Adam
        tuner = MyTuner(build_model_Adam,
                        objective=kt.Objective('val_mean_squared_error', direction='min'),
                        max_trials=n_trials,
                        directory='',
                        project_name=case_study,
                        overwrite=True)

    elif opt == 'SGD':
        # run model with SGD
        tuner = MyTuner(build_model_SGD,
                        objective=kt.Objective('val_mean_squared_error', direction='min'),
                        max_trials=n_trials,
                        directory='',
                        project_name=case_study,
                        overwrite=True)
                        
    elif opt == 'Nadam':
        # run model with Nadam
        tuner = MyTuner(build_model_Nadam,
                        objective=kt.Objective('val_mean_squared_error', direction='min'),
                        max_trials=n_trials,
                        directory='',
                        project_name=case_study,
                        overwrite=True)
    
    elif opt == 'Adagrad':
        # run model with Adagrad
        tuner = MyTuner(build_model_Adagrad,
                        objective=kt.Objective('val_mean_squared_error', direction='min'),
                        max_trials=n_trials,
                        directory='',
                        project_name=case_study,
                        overwrite=True)
    
    elif opt == 'RMSprop':
        # run model with RMSprop
        tuner = MyTuner(build_model_RMSprop,
                        objective=kt.Objective('val_mean_squared_error', direction='min'),
                        max_trials=n_trials,
                        directory='',
                        project_name=case_study,
                        overwrite=True)
    
    elif opt == 'Adamax':
        # run model with Adagrad
        tuner = MyTuner(build_model_Adamax,
                        objective=kt.Objective('val_mean_squared_error', direction='min'),
                        max_trials=n_trials,
                        directory='',
                        project_name=case_study,
                        overwrite=True)
    
    else:
        print ('Keras optimizer not defined!')
        sys.exit()

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=5)
    tuner.search(x_train, y_train, validation_data=(x_validation, y_validation), callbacks=[stop_early], verbose=0, use_multiprocessing=True, workers=14)

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters()[0]
    param_values = best_hps.values
    best_model = tuner.get_best_models()[0]
    #model = tuner.hypermodel.build(best_hps)
    
    return best_model, param_values