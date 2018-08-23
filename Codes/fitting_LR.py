from sklearn.linear_model import LogisticRegression

from kaggle_dataset import Preparing_Data as PD
X_train, Y_train, X_test = PD.X_train, PD.Y_train, PD.X_test
    
    
def LR_fit(c_value):
    #...Fitting the logistic regression
    clf = LogisticRegression(C = c_value)
    clf.fit(X_train,Y_train)
    
# =============================================================================
#     #...Finding the coefficients
#     coef_array = clf.coef_[0]
#     col_names = X_train.columns
#     for i in range(len(col_names)):
#         print col_names[i], coef_array[i]
# =============================================================================
    
    #...Finding the model output dependent feature
    Y_model_output = clf.predict(X_test)
    probability =  clf.predict_proba(X_test)
    probability_array = []
    for i in range(len(probability)):
        probability_array.append(probability[i][0])
    
    return probability_array, Y_model_output;