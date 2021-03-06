import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.spatial.distance import euclidean



def bray_curtis(a,b):
    
    return distance.braycurtis(a, b)


def camberra(a,b):

    return distance.canberra(a, a)


def Chebyshev(a,b):

    return distance.chebyshev(b, a)

def cityblock(a,b):

    return distance.cityblock(b, a)


def correlation(a,b):

    return distance.correlation(a, b)


def euclidian(a,b):

    return distance.euclidean(a, b)

def cosine(a,b):

    return distance.cosine(a, b) 


def hamming(a,b):

    return distance.hamming(a, b)


def jaccard(a,b):

    return distance.jaccard(a, b)

def gower(a,b):

    n = len(a)
    ans = 0
    
    for i in range(0, n):
        ans += np.abs(a[i] - b[i])
    
    return ans/n

def loren(a, b):
    n = len(a)
    ans = 0
    
    for i in range(0, n):
        op = np.abs(a[i] - b[i]) +1
        ans += np.log(op)
        
    return ans


def hellinger(P,Q):
    #P = P.tolist()
    #Q = Q.tolist()   
 
    diff = 0
    for i in range(0, len(P)):
        
        diff += (np.sqrt(P[i]) - np.sqrt(Q[i]))**2
    return 1/np.sqrt(2)*np.sqrt(diff)


def kullback_leibler(a,b):

    return entropy(a,b)


def waveHedges(a, b):
    

    n = len(a)
    ans = 0
    
    for i in range(0, n):
        ans += 1 - ( np.minimum(a[i], b[i]) / np.maximum(a[i], b[i]))
                
    return ans






def measure_distance(a, b, medida):

    medidas_x = ['gower', 'loren', 'hellinger', 'kullback_leibler', 'waveHedges']
    boo = medida in medidas_x


    if not boo:
        a = np.reshape(a, (1, a.shape[0]))
        b = np.reshape(b, (1, b.shape[0]))
    
    if medida == 'kullback_leibler':
        a = np.reshape(a, (a.shape[0],))


    if medida == 'bray_curtis':
        return bray_curtis(a,b)

    elif medida == 'camberra':
        return camberra(a,b)

    elif medida == 'Chebyshev':
        return Chebyshev(a,b)

    elif medida == 'cityblock':
        return cityblock(a,b)

    elif medida == 'correlation':
        return correlation(a,b)

    elif medida == 'euclidian':
        return euclidean(a,b) 

    elif medida == 'dtw':
        print('Distance Not Founded')
        return None

    elif medida == 'cosine':
        return cosine(a,b)

    elif medida == 'hamming':
        return hamming(a,b)

    elif medida =='jaccard':
        return jaccard(a,b)

    elif medida =='gower':
        return gower(a,b)

    elif medida == 'loren':
        return loren(a,b)

    elif medida == 'hellinger':
        return hellinger(a,b)

    elif medida == 'kullback_leibler':
        return kullback_leibler(a,b)

    elif medida == 'waveHedges':
        return waveHedges(a,b)

    elif medida == 'edrs':
        print('Not Founded')
        return None
    
    elif medida == 'shape_dtw':
        print('Not Founded')
        return None

    elif medida == 'sim_pocid':
        print('Not Founded')
        return None

    elif medida == 'pocid_mape':
        print('Not Founded')
        return None

    return None







    