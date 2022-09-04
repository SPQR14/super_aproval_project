

def histogram(df,col,bins,title):
    layout = go.Layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=30,xaxis= {"title": {"font": {"family": 'Courier New, monospace',"size": 18,
        "color": '#002e4d'}}},title_font_family="Courier New, monospace",title_font_color="#004878",template="plotly_white")
    fig=df[[col]].iplot(kind='histogram',x=col,bins=bins,title=title,asFigure=True,theme="white",layout=layout,color="#003e6c")
    fig.update_traces(opacity=0.90)
    return fig


def bar(df,col,title,x_title="",y_title=""):
    layout = go.Layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=30,xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 18,
        "color": '#002e4d'}}},yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 18,
        "color": '#002e4d'}}},title_font_family="Courier New, monospace",title_font_color="#004878",template="plotly_white")
    aux=pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={"index":"conteo"})
    fig=aux.iplot(kind='bar',x="conteo",y=col,title=title,asFigure=True,barmode="overlay",sortbars=True,color="#255479",layout=layout)
    fig.update_layout(width=800)
    fig.update_traces(marker_color='#005a96')
    return fig

def pie(df,col,title,x_title="",y_title=""):
    layout = go.Layout(template="plotly_white")
    colors=[ "#152337", "#183152","#17416d","#005096","#00569c","#005ba3","#0061a9","#1567af","#226cb6","#2c72bc", "#0061a9","#4c79b7","#7492c6","#98acd4","#bbc7e2","#dde3f1","#ffffff"]
    aux=pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={"index":"conteo"})
    fig=aux.iplot(kind='pie',labels="conteo",values=col,title=title,asFigure=True,theme="white")
    
    fig.update_traces(textfont_size=10,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=30,title_font_family="Courier New, monospace",title_font_color="#004878",template="plotly_white")
    return fig

def box(df,col,title):
    layout = go.Layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=30,xaxis= {"title": {"font": {"family": 'Courier New, monospace',"size": 18,
        "color": '#002e4d'}}},title_font_family="Courier New, monospace",title_font_color="#004878",template="plotly_white")
    fig=df[[col]].iplot(kind='box',title=title,asFigure=True,theme="white",layout=layout,color="#005a96", boxpoints='outliers')
    return fig