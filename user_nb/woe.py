import pandas as pd
import numpy as np


def woe(X_aux, feature, tgt):
    aux = X_aux[[feature, tgt]].groupby(feature).agg(["count", "sum"])
    aux["evento"] = aux[tgt, "sum"]
    aux["no_evento"] = aux[tgt, "count"] - aux[tgt, "sum"]
    aux["%evento"] = aux["evento"] / aux["evento"].sum()
    aux["%no_evento"] = aux["no_evento"] / aux["no_evento"].sum()
    aux["WOE"] = np.log(aux["%evento"] / aux["%no_evento"])
    IV=((aux["%evento"] - aux["%no_evento"])*aux["WOE"]).sum()
    aux.columns = aux.columns.droplevel(1)
    aux = aux[["WOE"]].reset_index().rename(columns={"WOE": f"W_{feature}"})
    return aux, IV
