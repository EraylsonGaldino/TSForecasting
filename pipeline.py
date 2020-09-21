import matplotlib.pyplot as plt
path = 'https://raw.githubusercontent.com/EraylsonGaldino/dataset_time_series/master/airline.txt'
import loadfiles as lf
import preprocessing as pre
import predictors as pred


def save_pickle_formart(model, window_size, h_step, training_percentage, validation_percentage, level_grid, filename_pickle = 'finalized_model.sav'):
    import pickle
    dict_pickle = {'model': model,
                  'window_size': window_size,
                  'h_step': h_step,
                  'training_percentage': training_percentage,
                  'validation_percentage': validation_percentage,
                  'testing_percentage': (1 - (training_percentage + validation_percentage)),
                  'level_grid': level_grid}


    pickle.dump(dict_pickle, open(filename_pickle, 'wb'))

def pipeline_1(path_data, window_size=20, h_step=1, training_percentage = 0.4, validation_percentage = 0.2, model='SVR', filename_pickle='SVR_HARD.sav', level_grid = 'hard'):
    data = lf.load_data_to_dataframe(path)
    serie = data[0].values
    normalized_data, scaler = pre.stand_interval(serie, 0, 1)
    sliding_windows = pre.create_windows_to_numpy(normalized_data, window_size, h_step) 
    
    if validation_percentage == 0.0:
        X_train, y_train, X_test, y_test = pre.split_series(sliding_windows, training_percentage, validation_percentage)
        trained_model = []
        if model == 'SVR':
            trained_model = pred.auto_SVR(X_train, y_train, level_grid = 'none')
        elif model == 'MLP':
            trained_model = pred.auto_MLP(X_train, y_train, level_grid = 'none')
        
        predict_test = trained_model.predict(X_test)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = pre.split_series(sliding_windows, training_percentage, validation_percentage)
        
        trained_model = []
        if model == 'SVR':
            trained_model = pred.auto_SVR(X_train, y_train, X_val, y_val, level_grid = level_grid)
        elif model == 'MLP':
            trained_model = pred.auto_MLP(X_train, y_train, X_val, y_val, level_grid = level_grid)
       
        predict_test = trained_model.predict(X_test)
        save_pickle_formart(model, window_size, h_step, training_percentage, validation_percentage, level_grid, filename_pickle)


    plt.clf() 
    plt.plot(y_test, label='target')
    plt.plot(predict_test,label ='predict')
    plt.show()





pipeline_1(path, training_percentage=0.4, validation_percentage=0.2)