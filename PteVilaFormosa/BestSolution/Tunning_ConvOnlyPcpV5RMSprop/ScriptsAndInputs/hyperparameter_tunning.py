from tensorflow.keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam
from keras.losses import mean_squared_error, mean_absolute_percentage_error
from keras.activations import relu, elu, sigmoid, softmax, softplus, softsign, tanh, selu, exponential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import tensorflow as tf
import kerastuner as kt

class MyTuner(kt.tuners.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
    
        # batch size and epochs
        kwargs['batch_size'] = hp.Int('batch_size', 10, 50, step=10)
        kwargs['epochs'] = hp.Int('epochs', 100, 400, step=50, default=150)

        super(MyTuner, self).run_trial(trial, *args, **kwargs)

def build_model_Adam(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    
    filters_value = hp.Choice('filter', values=[8, 16, 32])
    kernel_value = hp.Choice('kernel', values=[1, 5, 10])
    
    model.add(Conv1D(filters=filters_value, kernel_size=kernel_value, input_shape=(x_train_aux.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    
    # add 0, 1 or more convolutional layers
    n_conv_layers = hp.Choice('hidden_conv_layers', values=[0, 1, 2])
    if n_conv_layers != 0:
        filters_value_other_conv = hp.Choice('filter_others', values=[8, 16, 32])
        kernel_value_other_conv = hp.Choice('kernel_others', values=[8, 16, 32])
        
        model.add(Conv1D(filters=filters_value_other_conv, kernel_size=kernel_value_other_conv, padding='causal'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    add_dropout = hp.Boolean('add_drop', default=False)
    if n_hidden_layers != 0:
        nodes_l_hidden=hp.Choice('hidden_input_nodes', values=[3, 5, 10])
        hp_act_dense_other = hp.Choice('activation_input_den_hid', values=['softsign', 'linear', 'elu', 'relu'])
        dropout_value = hp.Choice('drop_value', values=[0.0, 0.1, 0.2])
        for layer in range(n_hidden_layers):
            model.add(Dense(units=nodes_l_hidden,activation=hp_act_dense_other))
            if add_dropout:
                model.add(Dropout(dropout_value))

    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'elu', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    hp_epsilon = hp.Choice('epsilon', values=[1e-7, 1e-8])
    model.compile(optimizer=Adam(lr=hp_learning_rate, epsilon=hp_epsilon, ), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_SGD(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    
    filters_value = hp.Choice('filter', values=[8, 16, 32])
    kernel_value = hp.Choice('kernel', values=[1, 5, 10])
    
    model.add(Conv1D(filters=filters_value, kernel_size=kernel_value, input_shape=(x_train_aux.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    
    # add 0, 1 or more convolutional layers
    n_conv_layers = hp.Choice('hidden_conv_layers', values=[0, 1, 2])
    if n_conv_layers != 0:
        filters_value_other_conv = hp.Choice('filter_others', values=[8, 16, 32])
        kernel_value_other_conv = hp.Choice('kernel_others', values=[8, 16, 32])
        
        model.add(Conv1D(filters=filters_value_other_conv, kernel_size=kernel_value_other_conv, padding='causal'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    add_dropout = hp.Boolean('add_drop', default=False)
    if n_hidden_layers != 0:
        nodes_l_hidden=hp.Choice('hidden_input_nodes', values=[3, 5, 10])
        hp_act_dense_other = hp.Choice('activation_input_den_hid', values=['softsign', 'linear', 'elu', 'relu'])
        dropout_value = hp.Choice('drop_value', values=[0.0, 0.1, 0.2])
        for layer in range(n_hidden_layers):
            model.add(Dense(units=nodes_l_hidden,activation=hp_act_dense_other))
            if add_dropout:
                model.add(Dropout(dropout_value))

    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'elu', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    model.compile(optimizer=SGD(lr=hp_learning_rate, momentum=0.9, nesterov=True), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_Nadam(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    
    filters_value = hp.Choice('filter', values=[8, 16, 32])
    kernel_value = hp.Choice('kernel', values=[1, 5, 10])
    
    model.add(Conv1D(filters=filters_value, kernel_size=kernel_value, input_shape=(x_train_aux.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    
    # add 0, 1 or more convolutional layers
    n_conv_layers = hp.Choice('hidden_conv_layers', values=[0, 1, 2])
    if n_conv_layers != 0:
        filters_value_other_conv = hp.Choice('filter_others', values=[8, 16, 32])
        kernel_value_other_conv = hp.Choice('kernel_others', values=[8, 16, 32])
        
        model.add(Conv1D(filters=filters_value_other_conv, kernel_size=kernel_value_other_conv, padding='causal'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    add_dropout = hp.Boolean('add_drop', default=False)
    if n_hidden_layers != 0:
        nodes_l_hidden=hp.Choice('hidden_input_nodes', values=[3, 5, 10])
        hp_act_dense_other = hp.Choice('activation_input_den_hid', values=['softsign', 'linear', 'elu', 'relu'])
        dropout_value = hp.Choice('drop_value', values=[0.0, 0.1, 0.2])
        for layer in range(n_hidden_layers):
            model.add(Dense(units=nodes_l_hidden,activation=hp_act_dense_other))
            if add_dropout:
                model.add(Dropout(dropout_value))

    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'elu', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    hp_epsilon = hp.Choice('epsilon', values=[1e-7, 1e-8])
    model.compile(optimizer=Nadam(lr=hp_learning_rate, epsilon=hp_epsilon, ), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_Adagrad(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    
    filters_value = hp.Choice('filter', values=[8, 16, 32])
    kernel_value = hp.Choice('kernel', values=[1, 5, 10])
    
    model.add(Conv1D(filters=filters_value, kernel_size=kernel_value, input_shape=(x_train_aux.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    
    # add 0, 1 or more convolutional layers
    n_conv_layers = hp.Choice('hidden_conv_layers', values=[0, 1, 2])
    if n_conv_layers != 0:
        filters_value_other_conv = hp.Choice('filter_others', values=[8, 16, 32])
        kernel_value_other_conv = hp.Choice('kernel_others', values=[8, 16, 32])
        
        model.add(Conv1D(filters=filters_value_other_conv, kernel_size=kernel_value_other_conv, padding='causal'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    add_dropout = hp.Boolean('add_drop', default=False)
    if n_hidden_layers != 0:
        nodes_l_hidden=hp.Choice('hidden_input_nodes', values=[3, 5, 10])
        hp_act_dense_other = hp.Choice('activation_input_den_hid', values=['softsign', 'linear', 'elu', 'relu'])
        dropout_value = hp.Choice('drop_value', values=[0.0, 0.1, 0.2])
        for layer in range(n_hidden_layers):
            model.add(Dense(units=nodes_l_hidden,activation=hp_act_dense_other))
            if add_dropout:
                model.add(Dropout(dropout_value))

    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'elu', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    hp_epsilon = hp.Choice('epsilon', values=[1e-7, 1e-8])
    model.compile(optimizer=Adagrad(lr=0.01, epsilon=hp_epsilon), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_RMSprop(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    
    filters_value = hp.Choice('filter', values=[8, 16, 32])
    kernel_value = hp.Choice('kernel', values=[1, 5, 10])
    
    model.add(Conv1D(filters=filters_value, kernel_size=kernel_value, input_shape=(x_train_aux.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    
    # add 0, 1 or more convolutional layers
    n_conv_layers = hp.Choice('hidden_conv_layers', values=[0, 1, 2])
    if n_conv_layers != 0:
        filters_value_other_conv = hp.Choice('filter_others', values=[8, 16, 32])
        kernel_value_other_conv = hp.Choice('kernel_others', values=[8, 16, 32])
        
        model.add(Conv1D(filters=filters_value_other_conv, kernel_size=kernel_value_other_conv, padding='causal'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    add_dropout = hp.Boolean('add_drop', default=False)
    if n_hidden_layers != 0:
        nodes_l_hidden=hp.Choice('hidden_input_nodes', values=[3, 5, 10])
        hp_act_dense_other = hp.Choice('activation_input_den_hid', values=['softsign', 'linear', 'elu', 'relu'])
        dropout_value = hp.Choice('drop_value', values=[0.0, 0.1, 0.2])
        for layer in range(n_hidden_layers):
            model.add(Dense(units=nodes_l_hidden,activation=hp_act_dense_other))
            if add_dropout:
                model.add(Dropout(dropout_value))

    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'elu', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    #hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    hp_epsilon = hp.Choice('epsilon', values=[1e-7, 1e-8])
    model.compile(optimizer=RMSprop(lr=0.01, epsilon=hp_epsilon), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def build_model_Adamax(hp):

    # next we can build the model exactly like we would normally do it
    model = Sequential()
    
    filters_value = hp.Choice('filter', values=[8, 16, 32])
    kernel_value = hp.Choice('kernel', values=[1, 5, 10])
    
    model.add(Conv1D(filters=filters_value, kernel_size=kernel_value, input_shape=(x_train_aux.shape[1], 1), padding='causal'))
    model.add(MaxPooling1D(pool_size=2))
    
    # add 0, 1 or more convolutional layers
    n_conv_layers = hp.Choice('hidden_conv_layers', values=[0, 1, 2])
    if n_conv_layers != 0:
        filters_value_other_conv = hp.Choice('filter_others', values=[8, 16, 32])
        kernel_value_other_conv = hp.Choice('kernel_others', values=[8, 16, 32])
        
        model.add(Conv1D(filters=filters_value_other_conv, kernel_size=kernel_value_other_conv, padding='causal'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # add 0, 1 or more dense layers as hidden layers
    n_hidden_layers = hp.Choice('hidden_n_layers', values=[0, 1, 2])
    add_dropout = hp.Boolean('add_drop', default=False)
    if n_hidden_layers != 0:
        nodes_l_hidden=hp.Choice('hidden_input_nodes', values=[3, 5, 10])
        hp_act_dense_other = hp.Choice('activation_input_den_hid', values=['softsign', 'linear', 'elu', 'relu'])
        dropout_value = hp.Choice('drop_value', values=[0.0, 0.1, 0.2])
        for layer in range(n_hidden_layers):
            model.add(Dense(units=nodes_l_hidden,activation=hp_act_dense_other))
            if add_dropout:
                model.add(Dropout(dropout_value))

    # then we finish again with completely standard Keras way
    hp_act_l3 = hp.Choice('activation_output', values=['softsign', 'linear', 'elu', 'relu'])
    model.add(Dense(1, activation=hp_act_l3))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    hp_epsilon = hp.Choice('epsilon', values=[1e-7, 1e-8])
    model.compile(optimizer=Adamax(lr=hp_learning_rate, epsilon=hp_epsilon, ), loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

def tuner(case_study, x_train, y_train, x_validation, y_validation, opt):

    tf.keras.backend.clear_session()
    
    global x_train_aux
    x_train_aux = x_train
    n_trials=500
    
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

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3)
    tuner.search(x_train, y_train, validation_data=(x_validation, y_validation), callbacks=[stop_early], verbose=0, use_multiprocessing=True, workers=10)

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters()[0]
    param_values = best_hps.values
    best_model = tuner.get_best_models()[0]
    #model = tuner.hypermodel.build(best_hps)
    
    return best_model, param_values