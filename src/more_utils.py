def info_df(df):
    """Function that returns size, datatype and length of a dataframe"""
    print("Nombre de lignes :", df.shape[0])  # Display the number of rows
    print("Nombre de colonnes :", df.shape[1])  # Display the number of columns
    print("----------------------")
    print(df.head())  # Display the 5 first rows
    print("----------------------")
    print(df.dtypes)  # Display column types
    print("----------------------")
    # Display the column names and their maximum length
    print("max len: ",
          dict([(v, df[v].apply(lambda r: len(str(r)) if r != None else 0).max()) for v in df.columns.values]))