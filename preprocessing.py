
### generate time windows 


def create_windows_to_dataframe2(series, window_size, forecasting_step):
    from pandas import concat

    cols, names = list(), list()

    for i in range(window_size, 0, -1):
        cols.append(series.shift(i))

    for i in range(0, forecasting_step):
        cols.append(series.shift(-i))

    agg = concat(cols, axis=1)
    agg.dropna(inplace=True)

    return agg


def create_list_of_sliding_windows(series, window_size, number_of_targets):
    """Create a list of sliding windows from a time series.
        Args:
            series (list): Time series
            window_size(int): Size of temporal windows
            number_of_targets(int): The number of outputs from the time window
        Returns:
            List of sliding windows. The list contains a set of sliding windows created according to the parameters
            window size and number of targets.

            For example, if the parameters are: series = [1, 2, 3, 4, 5, 6],
            window_size = 1 and number_of_targets = 1

            list_of_sliding_windows = [[1, 2], [2, 3], [4, 5], [5, 6]]
    """
    list_of_sliding_windows = []
    list_size_to_iterate = len(series) - window_size - number_of_targets
    for i in range(0, list_size_to_iterate):
        window = series[i: i + window_size]
        target = series[i + window_size:  i + number_of_targets + window_size]
        list_of_sliding_windows.append(window + target)

    return list_of_sliding_windows


def create_windows_to_dataframe(series, window_size, number_of_targets):
    """Create a data frame of sliding windows from a time series.
        Args:
            series (Data frame): Time series
            window_size(int): Size of temporal windows
            number_of_targets(int): The number of outputs from the time window
        Returns:
            Data frame of sliding windows. The DataFrame contains a set of sliding windows created according to the
            parameters window size and number of targets.

            For example, if the parameters are: series = [1, 2, 3, 4, 5, 6],
            window_size = 1 and number_of_targets = 1

            Dataframe_of_sliding_windows = [[1, 2], [2, 3], [4, 5], [5, 6]]
    """
    from pandas import DataFrame

    return DataFrame(
        create_list_of_sliding_windows(series.values.reshape(-1, ).tolist(), window_size, number_of_targets))


def create_windows_to_numpy(series, window_size, number_of_targets):
    """Create a Numpy of sliding windows from a time series.
        Args:
            series (np.array): Time series
            window_size(int): Size of temporal windows
            number_of_targets(int): The number of outputs from the time window
        Returns:
            Numpy of sliding windows. The Numpy contains a set of sliding windows created according to the parameters
            window size and number of targets.

            For example, if the parameters are: series = [1, 2, 3, 4, 5, 6],
            window_size = 1 and number_of_targets = 1

            Numpy_of_sliding_windows = [[1, 2], [2, 3], [4, 5], [5, 6]]
    """
    from numpy import array

    return array(create_list_of_sliding_windows(series.reshape(-1, ).tolist(), window_size, number_of_targets))


def round_numbers(number):
    """ Round a number.
        Args:
            number (float): Number to be rounded
        Returns:
            Rounded number. The number is rounded to its decimal place. If the decimal number is more than 0.5, the
            number is rounded upwards. Otherwise, it is rounded down.

            For example:
                7.6 is rounded to 8;
                7.50 is rounded to 8;
                7.49 is rounded to 7.
    """
    from math import floor, ceil
    if number - floor(number) < 0.5:
        return floor(number)
    return ceil(number)


def split_series(series, training_percentage, validation_percentage: float = 0.0):
    """ Function to divide the time series into subsamples.
        Args:
            series (list): Time series
            training_percentage(float): Percentage of the training sample.
            validation_percentage(float): Percentage of the validation sample.
        Returns:
            A sample divided into subsamples. If the value of the validation percentage is different from zero, three
            subsamples are returned: training, validation and testing. If not, just two subsamples: training and
            testing.
    """
    training_sample_size = round_numbers(len(series) * training_percentage)

    if validation_percentage > 0:
        validation_sample_size = round_numbers(len(series) * validation_percentage)

        training_sample = series[0:training_sample_size]
        validation_sample = series[training_sample_size:training_sample_size + validation_sample_size]
        testing_sample = series[(training_sample_size + validation_sample_size):]
        
        
        X_train = training_sample[:, 0: -1]
        y_train = training_sample[:, -1]
        X_val = validation_sample[:, 0:-1]
        y_val = validation_sample[:, -1]
        X_test = testing_sample[:, 0:-1]
        y_test = testing_sample[:, -1]
        

        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        training_sample = series[0:training_sample_size]
        testing_sample = series[training_sample_size:]
        
        
        X_train = training_sample[:, 0: -1]
        y_train = training_sample[:, -1]
        X_test = testing_sample[:, 0:-1]
        y_test = testing_sample[:, -1]

        return X_train, y_train, X_test, y_test
    
#### Standard
        
def resize_sample_to_2D(serie):
    import pandas as pd
    if(type(serie) == type(pd.DataFrame()) or type(serie) == type(pd.Series())):
        serie_np = serie.to_numpy()
        return serie_np.reshape(serie_np.shape[0], 1)
    else:
        return serie.reshape(serie.shape[0], 1)
    
    
def stand_interval(serie_real, min=0, max=1):
    
    
    '''
        input: serie numpy (n, )
        output:  serie numpy (n, ), scaler (MinMaxScaler object)
    '''
    
    from sklearn.preprocessing import MinMaxScaler
    
    if (len(serie_real.shape)==1):
        serie_real = resize_sample_to_2D(serie_real)
    
    scaler = MinMaxScaler(feature_range=(min, max)).fit(serie_real)
    serie_stand = scaler.transform(serie_real)
    return serie_stand, scaler
    
def stand_interval_inversed(serie_stand, scaler):
    inversed = scaler.inverse_transform(serie_stand)
    return inversed
    
    
def stand_mean(serie_real):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(serie_real)
    serie_stand = scaler.transform(serie_real)
    return serie_stand, scaler


def stand_log(serie_real):
    import numpy as np
    return np.log(serie_real)


def stand_log_inversed(serie_stand):
    import numpy as np
    return np.exp(serie_stand)
