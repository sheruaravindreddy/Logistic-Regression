import pandas as pd
import numpy as np
from fitting_LR import LR_fit


def finding_C_value(c_value):
    probability_array, Y_model_output = LR_fit(c_value)
    Y_actual_output = pd.read_csv('C:/Users/sheruaravindreddy/Downloads/Kaggle-data-set/gender_submission.csv')
    
    df = pd.DataFrame()
    df['predicted_output'] = pd.Series(Y_model_output)
    df[['actual_output']] = Y_actual_output[['Survived']]
    df['probability'] = pd.Series(probability_array) 
    
    pred_array = []
    total_correct = 0
    total_wrong = 0
    for i in range(len(Y_model_output)):
        if df['predicted_output'][i] == df['actual_output'][i]:
            pred_array.append(1)
            total_correct += 1
        else:
            pred_array.append(0)
            total_wrong += 1       
    df['correct_wrong'] = pd.Series(pred_array)
    
    df = df.sort_values('probability', ascending = [False])
    df = df.reset_index(drop=True)
    
    return df, total_correct, total_wrong;





start_c = 0.01
output_c_value = start_c
for c in np.arange(10**(-7),1,start_c):
    df, total_correct, total_wrong = finding_C_value(c)

    cum_correct = 0
    cum_wrong = 0
    max_KS = 0
    for i in range(len(df)):
        if df['correct_wrong'][i] == 1:
            cum_correct  += 1.0/total_correct * 100
        else:
            cum_wrong += 1.0/total_wrong * 100
        KS = cum_correct - cum_wrong
        if KS >= max_KS:
            max_KS = KS
    print max_KS,  c





