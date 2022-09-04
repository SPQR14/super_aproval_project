from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import chisquare
from sklearn.manifold import MDS
import pandas as pd
import numpy as np

def woe(df, columns, tar):
    
    '''This function returns a dataframe with the transformed categorical variables using WoE.
        1. df: pandas dataframe where the variables are.
        2. columns: python list, containing the names of the vairables that are going to be  transformed.
        3. tar: pythoon strign containing the name of the target variable. '''
    
    df[columns].fillna('Missings',inplace=True)

    for v in columns:
        aux = df[[tar,v]].pivot_table(index=v,columns=tar,aggfunc='size')
        woe = aux.apply(lambda x: x/sum(x)).apply(lambda x: np.log(x[1]/x[0]),axis=1)
        aux['woe'] = woe
        aux[v] = aux.index
        df[v + '_WoE'] = df[v].map(dict(zip(aux[v], aux['woe'])))
        
    return(df)

def stc(df):
    
    '''This simple function computes the reescalation of a pandas datafrmae, and returns a pandas
        dataframe reescaled, using StandardScaler.
        1. df: pandas dataframe, is the space to be transformed.'''
    
    mm = StandardScaler()
    return(pd.DataFrame(mm.fit_transform(df), columns = df.columns))

def pca(df, tresh = None, components = 2):
    
    '''This function computes and return pca for a matrix X, and returns the new Xp and the pca adjusted object.
        1. df: pandas dataframe, this dataframe is the matrix X.
        2. tresh: python float, indicating the quantity of variance that must be explained by the pca process. 0<tresh<1.
        3. componentes: python int, indicating the number of components to be computed.'''
    
    xs = stc(df)
    
    if tresh:
        v = 0
        k = components
        while v<tresh:
            print(f'Testing for k = {k}')
            pca = PCA(n_components = k)
            pca.fit(xs)
            v = sum(pca.explained_variance_ratio_)
            print(f'The cummuled explained variance for k={k} is: {v}')
            k += 1
        xp = pd.DataFrame(pca.transform(xs), columns = [f'P_{i}' for i in range(k-1)])
        return(xp, pca)
    
    else:
        k = components
        pca = PCA(n_components = k)
        pca.fit(xs)
        xp = pd.DataFrame(pca.transform(xs), columns = [f'P_{i}' for i in range(k)])
        return(xp, pca)

def mm(df):
    
    '''This simple function computes the reescalation of a pandas datafrmae, and returns a pandas
        dataframe reescaled, using MinMaxScaler.
        1. df: pandas dataframe, is the space to be transformed.'''
    
    mm = MinMaxScaler()
    r = pd.DataFrame(mm.fit_transform(df), columns = df.columns)
    return(r)


def mds(df, n_components = 2, mm_scaler = True):
    
    '''This function returns a new space X_mds, based on X, using the mds algorithm. It returns just
        X_mds pandas dataframe. The parameters are:
        1. df: Pandas dataframe containing the matrix of space X.
        2. n_components: python int, the number of components.
        3. mm_scaler: python boolean, it specifies if the MinMaxScaler transform is going to be required.''' 
    
    if mm_scaler:
        xmm = mm(df)
    else:
        xmm = df.copy()
    
    mds_ = MDS(n_components)
    x_mds = pd.DataFrame(mds_.fit_transform(xmm), columns = [f'd{i}' for i in range(n_components)])
    
    return(x_mds) 