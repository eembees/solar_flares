import numpy as np
import pandas as pd


def save_ys_csv(labels, ids=None):
    if ids == None:
        ids = np.arange(1,len(labels)+1, dtype=int)

    df = pd.DataFrame({'Id':ids,'ClassLabel':labels})
    df.to_csv('./output/test.csv')