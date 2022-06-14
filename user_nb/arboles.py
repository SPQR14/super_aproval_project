
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def cuts(df,feature,tgt):
    df[feature]=df[feature].astype(float)
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.08, max_features=1,random_state=0)
    X = df.loc[df[feature].notnull(), [feature]]
    y = df.loc[df[feature].notnull(), tgt]
    dt.fit(X, y)
    df.loc[df[feature].notnull(), f"aux_{feature}"] = dt.apply(X)
    aux = df[[f"aux_{feature}", feature]].groupby([f"aux_{feature}"]).agg(["min", "max"])
    aux.columns = aux.columns.droplevel(0)
    aux[f"Interval_{feature}"]= aux.apply(lambda x:[x['min'],x['max']],axis=1)
    inter_list=aux[f"Interval_{feature}"].tolist()
    inter_list[0][0]=-np.Inf
    inter_list[-1][-1]=np.Inf
    inter_list=pd.IntervalIndex.from_tuples(list(map(tuple,inter_list)),closed="both")
    name=feature.replace("c_","")
    df[f"v_arbol_{name}"]=pd.cut(df[feature].astype(float),bins=inter_list)
    df.drop(columns=[f"aux_{feature}"],inplace=True)
    return df

def tree_cut(X_tr, X_te, feature, new_category, tgt, max_depth, min_samples_leaf):
    X_train = X_tr.copy()
    X_test = X_te.copy()

    # Árbol de decisión
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=1,random_state=0)
    X_feat = X_train.loc[X_train[feature].notnull(), [feature]]
    y_feat = X_train.loc[X_train[feature].notnull(), tgt]
    dt.fit(X_feat, y_feat)
    X_train.loc[X_train[feature].notnull(), new_category] = dt.apply(X_train.loc[X_train[feature].notnull(), [feature]])
    X_test.loc[X_test[feature].notnull(), new_category] = dt.apply(X_test.loc[X_test[feature].notnull(), [feature]])

    #intervales
    aux_train = X_train[[new_category, feature]].groupby([new_category]).agg(["min", "max"])

    aux_train.columns = aux_train.columns.droplevel(0)

    aux_train[f"Interval_{feature}"]= aux_train.apply(lambda x:[x['min'],x['max']],axis=1)
    inter_list_train=aux_train[f"Interval_{feature}"].tolist()
    inter_list_train[0][0]=-np.Inf
    inter_list_train[-1][-1]=np.Inf

    inter_list_train=pd.IntervalIndex.from_tuples(list(map(tuple,inter_list_train)),closed="both")

    X_train[new_category]=pd.cut(X_train[feature].astype(float),bins=inter_list_train)
    X_test[new_category]=pd.cut(X_test[feature].astype(float),bins=inter_list_train)

    #Revisamos si hay nulos
    if (X_train[new_category].isnull().sum() > 0):
        X_train[new_category] = X_train[new_category].cat.add_categories('missings')
        X_train[new_category].fillna('missings', inplace =True)

        X_test[new_category] = X_test[new_category].cat.add_categories('missings')
        X_test[new_category].fillna('missings', inplace =True)

    return X_train,X_test
