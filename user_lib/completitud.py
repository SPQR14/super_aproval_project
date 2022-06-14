import pandas as pd

def completitud(df):
    completitud = pd.DataFrame(df.isnull().sum())
    completitud.reset_index(inplace=True)
    completitud=completitud.rename(columns={'index':'columna', 0:'total faltantes'})
    completitud['completitud']=(1-completitud['total faltantes']/df.shape[0]) * 100
    completitud=completitud.sort_values(by='completitud', ascending=True)
    return completitud