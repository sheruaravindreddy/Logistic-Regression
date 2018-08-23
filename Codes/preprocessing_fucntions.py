def drop_col(df):
#    df = df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age'])
    return df;

def label_encoder(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df = df.apply(le.fit_transform)
    return df;

def scale(df):
    from sklearn.preprocessing import scale
    df = scale(df)
    return df;

def min_max_scaler(df):
    from sklearn.preprocessing import MinMaxScaler
    min_max = MinMaxScaler()
    df = min_max.fit_transform(df)
    return df;

def normalize(df):
    from sklearn.preprocessing import normalize
    df = normalize(df, norm = 'l2')
    return df;