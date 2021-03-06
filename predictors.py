def fit_sklearn(model, X_train, y_train):
    return model.fit(X_train, y_train)


def predict_sklearn(model, X):
    return model.predict(X)


def generate_SVR(k='rbf', g=0.1, e=0.01, c=4):
    from sklearn.svm import SVR
    return SVR(kernel=k, gamma=g, epsilon=e, C=c)


def generate_MLP(h_layer_s=100, act='relu', sv='adam', max_it=200, learning_rate='constant'):
    from sklearn.neural_network import MLPRegressor
    return MLPRegressor(hidden_layer_sizes=h_layer_s, activation=act, solver=sv, max_iter=max_it,
                        learning_rate='constant')


def generate_TREE(max_depth=10, criterion="mse"):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=max_depth, criterion=criterion)


def auto_SVR(X_train, y_train, X_val=[], y_val=[], level_grid='easy'):
    """ Train the SVR with values pre defined to hyper-param

    perc_val: define the percentual value to create de validation sample
    level_grid: difine the number of values of each hyper-param
              values: 'easy', 'medium', 'hard'
    """
    from sklearn.metrics import mean_squared_error as MSE
    import numpy as np
    import itertools

    kernel = gamma = eps = C = 0
    if level_grid == 'none':
        svr = generate_SVR()
        svr = fit_sklearn(svr, X_train, y_train)
        return svr
    elif level_grid == 'easy':
        kernel = ['rbf']
        gamma = [0.1, 1]
        eps = [0.1, 0.001]
        C = [0.1, 10]
    elif level_grid == 'medium':
        kernel = ['rbf', 'sigmoid']
        gamma = [0.1, 1, 1000, 10000]
        eps = [0.01, 0.001, 0.0001, 0.000001]
        C = [0.1, 1, 1000]
    elif level_grid == 'hard':
        kernel = ['rbf', 'sigmoid', 'poly']
        gamma = [0.001, 0.1, 1, 1000, 10000]
        eps = [0.01, 0.001, 0.0001, 0.000001]
        C = [0.1, 1, 1000]

    hyper_param = list(itertools.product(kernel, gamma, eps, C))
    best_result = np.Inf
    select_model = generate_SVR()
    print('training the SVR with ', level_grid, ' gridsearch')
    for k, g, e, c in hyper_param:
        svr = generate_SVR(k, g, e, c)
        svr = fit_sklearn(svr, X_train, y_train)
        predict_val = predict_sklearn(svr, X_val)
        mse_val = MSE(y_val, predict_val)

        if mse_val < best_result:
            best_result = mse_val
            select_model = svr

    return select_model


def auto_MLP(X_train, y_train, X_val=[], y_val=[], level_grid='easy'):
    """
    train the MLP with values pre defined to hyper-param

    perc_val: define the percentual value to create de validation sample
    level_grid: difine the number of values of each hyper-param
              values: 'easy', 'medium', 'hard'
  
    
    """

    from sklearn.metrics import mean_squared_error as MSE
    import numpy as np
    import itertools

    hidden_layer_sizes = activation = solver = max_iter = learning_rate = []
    if level_grid == 'none':
        mlp = generate_MLP()
        mlp = fit_sklearn(mlp, X_train, y_train)
        return mlp
    elif level_grid == 'easy':
        hidden_layer_sizes = [1, 100]
        activation = ['identity', 'logistic']
        solver = ['lbfgs', 'adam']
        max_iter = [1000]
        learning_rate = ['constant', 'adaptive']
    elif level_grid == 'medium':
        hidden_layer_sizes = [1, 25, 50, 100]
        activation = ['identity', 'logistic']
        solver = ['lbfgs', 'sgd', 'adam']
        max_iter = [1000]
        learning_rate = ['invscaling', 'adaptive']
    elif level_grid == 'hard':
        hidden_layer_sizes = [1, 5, 10, 50, 100]
        activation = ['identity', 'tanh', 'relu', 'logistic']
        solver = ['lbfgs', 'sgd', 'adam']
        max_iter = [1000]
        learning_rate = ['constant', 'invscaling', 'adaptive']

    hyper_param = list(itertools.product(hidden_layer_sizes, activation, solver, max_iter, learning_rate))
    best_result = np.Inf
    select_model = generate_MLP()
    print('training the MLP with ', level_grid, ' gridsearch')

    for hls, a, s, mi, lr in hyper_param:
        mlp = generate_MLP(hls, a, s, mi, lr)
        mlp = fit_sklearn(mlp, X_train, y_train)
        predict_val = predict_sklearn(mlp, X_val)
        mse_val = MSE(y_val, predict_val)

        if mse_val < best_result:
            best_result = mse_val
            select_model = mlp

    return select_model


if __name__ == '__main__':
    print('predictors...')
