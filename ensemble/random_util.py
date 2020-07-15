import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def multi_class_F1score(val_df, cost_file, mode='af'):
    
    y_val_df = pd.read_csv('./SF/dist_1/SF_val/y_true.txt', header=None)
    y_true = np.array(y_val_df)
    
    cost_df = pd.read_csv('./cost_matrix/{}.txt'.format(cost_file), header=None)
    cost_arr = np.array(cost_df)
    if mode == 'af':
        cost_arr = 1
        
    y_pred = np.argmax(np.array(val_df), 1)
    val_cm = confusion_matrix(y_true, y_pred)
     
    G_matrix = val_cm * cost_arr
    
    score_idx = np.array([]) 
    row, col = G_matrix.shape
    numsam_perclass = np.sum(G_matrix, 1)

    for i in range(row):
        recall = G_matrix[i, i] / np.sum(G_matrix[i, :])
        precision = G_matrix[i, i] / np.sum(G_matrix[:, i])
        class_score = 2 * (precision * recall) / (precision + recall)
        score_idx = np.append(score_idx, class_score)
    F1_score = np.sum((numsam_perclass * score_idx)) / np.sum(G_matrix)
    return F1_score


def minmax(array, Min, Max):
    
    row, col = array.shape
    output = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            scale = (array[i, j] - array[i, :].min()) * (Max - Min) / (array[i, :].max() - array[i, :].min()) + Min
            output[i, j] = scale
    _, lens = output.shape 
    output = np.reshape(output, (lens, 1))
    return output

def sens_spec(cm):
    sens = np.zeros((1,8))
    spec = np.zeros((1,8))
    TP_all = 0
    for i in range(8):
        TP_all = TP_all + cm[i,i]
    for i in range(8):
        sens[0,i] = cm[i,i] / cm[i,:].sum()
        spec[0,i] = (cm.sum()-cm[i,:].sum()-cm[:,i].sum()+cm[i,i]) / (cm.sum()-cm[i,:].sum())

    spec = np.reshape(spec, (8,1))
    return sens, spec

def total_cost(cm, cost_file):
    
    cost_df = pd.read_csv('./cost_matrix/{}.txt'.format(cost_file), header=None)
    cost_arr = np.array(cost_df)
    over_all_cost = np.sum(cm * cost_arr)
    return over_all_cost
    

