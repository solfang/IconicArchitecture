import pandas as pd

def create_pivot_table(df, row_label, column_label, value_label):
    """
    Creates a pivot table in this format:
                col1     col2
    row1
    row2       ...values...
    row3

    This won't work if row_label or column_label is the same as value_label (e.g. you only have 2 distinct labels, not 3).
    In that case add a dummy column that is a duplicate of the column

    :param df: pandas dataframe
    :param row_label: column whose unique values will be the rows of the pivot tables
    :param column_label: column whose unique values will be the columns of the pivot tables
    :param value_label: column that contains values to be counted
    :return: pivot table
    """
    grouped = df.groupby([row_label, column_label])[value_label].count().reset_index()
    piv = grouped.pivot(index=row_label, columns=column_label, values=value_label)
    return piv