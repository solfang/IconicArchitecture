import pandas as pd
import numpy as np
import os

def summarize(df, do_print=True, fpath=None, shape=True, head=True, describe=True, values=True):

    s = ""
    if fpath:
        s += "Saved to {}\n\n".format(fpath)

    s += "# Summary"
    if shape:
        s += "\n\n## Shape\nObservations: {}\nFeatures: {}".format(df.shape[0], df.shape[1])
        s+= "\nColumns:\n{}".format(list(df.columns))

    if head:
        s += "\n\n## Head\n {}".format(df.head().to_string())

    if describe:
        s += "\n\n## Describe\n{}".format(df.describe().round(1).to_string())

    if values:
        s += "\n\n## Values"
        s += "\nColumn         Values        Missing  Unique Values"
        s += "\n-----------------------"
        for column in df.columns:
            missing = df[column].isna().sum()
            values = len(df) - missing
            unique = df[column].apply(
                str).unique()  # converting everything to string so we can do unique() on iterables like lists etc.
            unique_sorted = sorted(unique)
            unique_str = "({} unique values)".format(len(unique)) if len(unique) > 20 else str(unique_sorted)
            s += "\n{} {} \t\t {} \t\t {}".format(column.ljust(15), values, missing, unique_str)

    if do_print:
        print(s)
    if fpath:
        os.makedirs(os.path.dirname(fpath),exist_ok=True)
        with open(fpath, 'w') as f:
            print(s, file=f)