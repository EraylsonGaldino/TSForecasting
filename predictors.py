import numpy as np


def fit_sklearn(model, X_train, y_train):
    
    return model.fit(X_train, y_train)
	

def predict_sklearn(model, X):
    
    return model.predict(X)


def generate_SVR(k='rbf', g=0.1, e=0.01, c=4):
	from sklearn.svm import SVR
	return SVR(kernel=k,gamma=g, epsilon=e, C=c )


def generate_MLP(h_layer_s = 100, act = 'relu', sv = 'adam', max_it=200):
	from sklearn.neural_network import MLPRegressor
	return MLPRegressor(hidden_layer_sizes= h_layer_s, activation=act, solver=sv, max_iter = max_it)
	
	 


def generate_TREE(max_depth = 10, criterion="mse"):
    
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(max_depth=max_depth, criterion = criterion)


def auto_SVR(X_train, y_train, perc_val = 0.2):
    from sklearn.metrics import mean_squared_error as MSE
    
    
    kernel =  ['rbf'] #['rbf', 'sigmoid', 'poly']
    gamma =  [0.5, 10] # [0.5, 1, 10, 100, 1000]
    eps = [1, 0.1] # [1, 0.1,0.001, 0.0001, 0.00001, 0.000001]
    C =  [1, 10] #[0.1, 1,  10, 100, 1000]
    
    size_val = int(np.fix(len(y_train)))
    X_val = X_train[-size_val:, :]
    y_val = y_train[-size_val:]
    X_train = X_train[0:size_val, :]
    y_train = y_train[0:size_val]
    
    best_result = np.Inf

    for k in kernel:
        for g in gamma:
            for e in eps:
                for c in C:
                    svr = generate_SVR(k,g,e, c)
                    svr = fit_sklearn(svr, X_train, y_train)
                    predict_val = predict_sklearn(svr, X_val)
                    mse_val = MSE(y_val, predict_val)
                    
                    
                    if mse_val < best_result:
                        best_result = mse_val
                        select_model = svr
                        
    return select_model
    
    











