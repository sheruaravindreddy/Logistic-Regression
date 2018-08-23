import pandas as pd
import numpy as np
from fitting_LR import LR_fit
# =============================================================================
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('five-thirty-eight')
# =============================================================================

ROC_df = pd.DataFrame(columns = ['c_value', 'TPR', 'FPR'])

def finding_C_value(c_value):
    Y_model_output = LR_fit(c_value)
    Y_actual_output = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/gender_submission.csv')
    
    df = pd.DataFrame()
    df['predicted_output'] = pd.Series(Y_model_output)
    df[['actual_output']] = Y_actual_output[['Survived']]
    
    TP, TN, FP, FN = 0.0,0.0,0.0,0.0
    for i in range(len(df)):
        if df['predicted_output'][i] == df['actual_output'][i] == 1:
            TP += 1
        elif df['predicted_output'][i] == df['actual_output'][i] == 0:
            TN += 1
        elif (df['predicted_output'][i] == 0) and (df['actual_output'][i] == 1):
            FP += 1
        elif (df['predicted_output'][i] == 1) and (df['actual_output'][i] == 0):
            FN += 1
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    F1_score = 2*precision*recall/(precision + recall)
    
    TPR = recall
    FPR = 1 - FP/(FP+TN)
    temp_df = pd.DataFrame({'c_value':c_value,'TPR':TPR,'FPR':FPR}, index=[0])
    global ROC_df
    ROC_df = ROC_df.append(temp_df,ignore_index = True,sort = False)
#    print "TP is %d and TN is %d and FP is %d and FN is %d"%(TP,TN,FP,FN)
#    print precision, recall, accuracy, F1_score
    return c_value, accuracy,F1_score;





max_accuracy = 0
start_c = 0.01
output_c_value = start_c
for c in np.arange(10**(-7),1,start_c):
    c_value, accuracy, F1_score = finding_C_value(c)
    if max_accuracy < accuracy:
        max_accuracy = accuracy
        output_c_value = c_value
#        print max_accuracy,output_c_value

#print ROC_df

import plotly
import plotly.graph_objs as go

plotly.offline.plot({
    "data": [go.Scatter(x=ROC_df.FPR, y=ROC_df.TPR)],
    "layout": go.Layout(title="ROC Curve",xaxis = dict(range = [0,1]),yaxis = dict(range = [0,1]))
}, auto_open=True)