import pandas as pd

class Preparing_Data:
    from preprocessing_fucntions import label_encoder,drop_col
    
    #...Importing the training dataset
    df_train = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/train.csv')
    Y_train = df_train[['Survived']]
    X_train = df_train.drop(columns = ['Survived'])
    #...Importing the testing dataset
    X_test = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/test.csv')
    
    #...LabelEncoding the dataset
    X_train = label_encoder(X_train)
    X_test = label_encoder(X_test)
    
# =============================================================================
#     #...1.Scale
#     X_train = pf.scale(X_train)
#     X_test = pf.scale(X_test)
# =============================================================================
    
# =============================================================================
#     #...2.Min_max_scaling
#     X_train = pf.min_max_scaler(X_train)
#     X_test = pf.min_max_scaler(X_test)
# =============================================================================
    
# =============================================================================
#     #...3.Normalize
#     X_train = pf.normalize(X_train)
#     X_test = pf.normalize(X_test)
# =============================================================================
    
# =============================================================================
#     #...Dropping few columns
#     X_train = drop_col(X_train)
#     X_test = drop_col(X_test)
# =============================================================================
