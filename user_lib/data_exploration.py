import plotly
import chart_studio.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.express as px


from plotly.graph_objs import Scatter, Figure, Layout
from plotly import tools

def histograms(df,list_v,frac=.3):
    
    #returns the histograms of the variables 
    
    for var in list_v:
        fig = px.histogram(df.sample(frac=frac), x=var,color_discrete_sequence= px.colors.sequential.Inferno,
                          title=f'Hist of {var}',)
        fig.show()
    return()


def iqr_(df, variables, alpha = 1):

    
    for v in variables:
        q3 = df[v].quantile(.75)
        q1 = df[v].quantile(.25)        
        iqr = q3 - q1
        lb, up = q1-(alpha*iqr), q3+(alpha*iqr)
        df = df.loc[(df[v]>=lb) & (df[v]<=up)].copy()
        
    return(df)   