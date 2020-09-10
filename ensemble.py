# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:39:50 2020

@author: pesqu
"""

import numpy as np
import predictors as p
from sklearn.metrics import mean_squared_error as MSE

def reamostragem(serie, n):
    size = len(serie)
    #nova_particao = []
    ind_particao = []
    for i in range(n):
        ind_r = np.random.randint(size)
        ind_particao.append(ind_r)
        #nova_particao.append(serie[ind_r,:])
    
    return ind_particao



def bagging(qtd_modelos, X_train, y_train):
    
    
    ens = []
    ensemble = {'models':[], 'indices': [] }   
    ind_particao = []
    
    if len(y_train.shape) == 1:
        y_train =  y_train.reshape(len(y_train), 1)
        
        
       
	
    train = np.hstack([X_train, y_train])
    
    for i in range(qtd_modelos):
        
        print('Training model: ', i)
        tam = len(y_train)
        indices = reamostragem(train, tam)
        
        particao = train[indices, :]
        
        
        X_train, y_train = particao[:, 0:-1], particao[:, -1]
        
        model = p.auto_SVR(X_train, y_train)
        #return modelo
        ens.append(model)
        ind_particao.append(indices)
        
    
    
    ensemble['models'] = ens
    ensemble['indices'] = ind_particao
    
   
    return ensemble




def dynamic_predict_selection(window_test_inst= [], X_train = [], y_train = [], ensemble = [], k = 1, approach='ola', metrica_ola='euclidian', X_prev = [], y_prev = []):
    
    
    if approach == 'ola':
        model_selected, ind_best = selec_model_ola(window_test_inst, X_train, y_train, ensemble, k, metrica='euclidian')
        return model_selected, ind_best
    
    if approach == 'dsnaw':
        model_selected, ind_best = nearest_antecedent_windows(X_prev, y_prev, ensemble)
        return model_selected, ind_best
    
    
def dynamic_ensemble_selection(n=10, window_test_inst= [], X_train = [], y_train = [], ensemble = [], k = 1, approach='ola', metrica_ola='euclidian', X_prev = [], y_prev = []):
    
    ensemble_selected = []
    if approach == 'ola':
        
        erros = selec_model_ola_erro(window_test_inst, X_train, y_train, ensemble, k, metrica_ola)
        
    elif approach == 'dsnaw':
        
        erros = nearest_antecedent_windows_error(X_prev, y_prev, ensemble)
        
    
    models_sorted = select_model_less(erros)
    
    for i in range(n):
        ensemble_selected.append(ensemble['models'][models_sorted[i]])
        
    return ensemble_selected
        
        


def selec_model_ola_erro(window_test_inst, X_train, y_train, ensemble, k, metrica='euclidian'):
    from similarity_measures import measure_distance
    #  Seleciona o modelo baseado no desempenho obtido em prever as k janelas mais próximas da nova janela    

    
    dist = []
    for i in range(0,len(X_train)):
        #d = euclidean(window_test_inst, x_data[i,:])
        d = measure_distance(window_test_inst, X_train[i,:], metrica)
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(X_train))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    #print(indices_patterns)
    indices_patterns_l = list(indices_patterns)
 
    k_patterns_x = X_train[indices_patterns_l[0:k]]
    k_patterns_y = y_train[indices_patterns_l[0:k]]
    
    
    
    error = []
    for i in range(0, len(ensemble['models'])):
        model = ensemble['models'][i]
        
        
        
        prev = model.predict(k_patterns_x)
 
        er = 0
        if len(prev) > 1:
            er = MSE(k_patterns_y, prev)
        else:
            er = np.absolute(k_patterns_y - prev)            
        
        
        error.append(er)
               
    return error


    


def selec_model_ola(window_test_inst, X_train, y_train, ensemble, k, metrica='euclidian'):
    from similarity_measures import measure_distance
    #  Seleciona o modelo baseado no desempenho obtido em prever as k janelas mais próximas da nova janela    
    #max_lag = len(lags_acf)
    x_data = X_train
    y_data = y_train
    
    

    #tam = len(x_data[0]) 
    
    dist = []
    for i in range(0,len(x_data)):
        #d = euclidean(window_test_inst, x_data[i,:])
        d = measure_distance(window_test_inst, x_data[i,:], metrica)
        #print(d)
        dist.append(d)
        
    indices_patterns = range(0, len(x_data))
    
    dist, indices_patterns = zip(*sorted(zip(dist, indices_patterns))) #returna tuplas ordenadas
    #print(indices_patterns)
    indices_patterns_l = list(indices_patterns)
 
    k_patterns_x = x_data[indices_patterns_l[0:k]]
    k_patterns_y = y_data[indices_patterns_l[0:k]]
    
    
    
    best_result = np.Inf
    for i in range(0, len(ensemble['models'])):
        model = ensemble['models'][i]
        
        
        
        prev = model.predict(k_patterns_x)
        mse = MSE(k_patterns_y, prev)
        #print('MSE do modelo', i,':', mse)     
        
        if mse < best_result:
            best_result = mse
            select_model = model
            ind_best = i
            
    
    return select_model, ind_best


def nearest_antecedent_windows_error(X_previous, y_previous, ensemble):

    '''
    Seleciona o modelo baseado em quem foi melhor nos pontos anteriores: BPM

    '''

    error = [] 
    for i in range(0, len(ensemble['models'])):
        model = ensemble['models'][i]
        
        
        prev = model.predict(X_previous)
        er = 0
        if len(prev) > 1:
            er = MSE(y_previous, prev)
        else:
            er = np.absolute(y_previous - prev)            
        
        
        error.append(er)
               
    return error


def select_model_less(valores_modelos):

    #seleciona o modelo com menor valor
    
    indices = range(0, len(valores_modelos))
    
    valores_ordenados, indices = zip(*sorted(zip(valores_modelos, indices))) #returna tuplas ordenadas
    
    return indices



def nearest_antecedent_windows(X_previous, y_previous, ensemble):
    
    #max_lag = len(lags_acf)

    
    best_result = np.Inf
    for i in range(0, len(ensemble['modelos'])):
        model = ensemble['modelos'][i]
        
        
        
        prev = model.predict(X_previous)
        
        if len(prev) > 1:
            error = MSE(y_previous, prev)
        else:
            error = np.absolute(y_previous - prev)
        #print('MSE do modelo', i,':', mse)     
        
        if error < best_result:
            best_result = error
            select_model = model
            ind_best = i
            
                  
                  
    return select_model, ind_best



def combination(ensemble, window_test_inst, approach = 'mean'):
    
    predictions = []
    
    for model in ensemble:
        pred = model.predict(window_test_inst)
        predictions.append(pred)
        
        
    if approach == 'mean':
        return np.mean(predictions)
    
    if approach == 'median':
        return np.median(predictions)
        
        
        
    