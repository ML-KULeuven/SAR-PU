# Pandas DF  manipulations
def binarize(df,column):
    column_values = set(df[column])
    df_binarized = df.drop(column,axis=1)
    for val in column_values:
        df_binarized[column+"-"+str(val)]=2*(df[column]==val)-1
    return df_binarized

def keep_k_most_common(df, column, k, other="other"):

    to_keep = df[column].value_counts()[:k].index.tolist()
    df_new = df.copy()
    df_new[column]=other
    for v in to_keep:
        df_new.loc[df[column]==v, column]=v
    return df_new
