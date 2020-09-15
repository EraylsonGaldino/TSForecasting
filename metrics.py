import numpy as np

def error(target, predicted):
    """ Error (target - predicted)  """

    if type(target) != type(np.array([])):
        target = np.array(target)

    if type(predicted) != type(np.array([])):
        predicted = np.array(predicted)
    
    return target - predicted

def mse(target, predicted):
    """ Mean Squared Error """
    return np.mean(np.square(error(target, predicted)))


def rmse(target, predicted):
    """ Root Mean Squared Error """
    return np.sqrt(mse(target, predicted))


def mae(target, predicted):
    """ Mean Absolute Error """
    return np.mean(np.abs(error(target, predicted)))

def mape(target, predicted):
    """ Mean absolute percentage error """
  
    size = len(target)
    sum_error = 0
    for i in range(0, len(target)):
        if target[i] > 0.0:
            x = np.abs((predicted[i] - target[i]) / target[i])

        else:
            x = 0.0
        
        sum_error = x + sum_error
    return (100/size) * sum_error



if __name__ == "__main__":
    target = [1, 1, 1, 1]
    predicted = [1, 1, 1, 1]
    print(mape(target, predicted))