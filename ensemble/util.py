import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def compute_acc(y_pred):
    y_df = pd.read_csv('./SF/y_true.txt', header=None)
    y_true = np.array(y_df)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def multi_class_F1score(mode, val_df):

    if mode == 'cost':
        cost_df = pd.read_csv('./cost_matrix/cost_matrix.txt', header=None)
    else:
        cost_df = pd.read_csv('./cost_matrix/saline.txt', header=None)

    cost_arr = np.array(cost_df)
    val_arr = np.array(val_df)
    G_matrix = val_arr * cost_arr
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

def total_cost(y_pred):
    y_df = pd.read_csv('./SF/y_true.txt', header=None)
    y_true = np.array(y_df)
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cost_M = pd.read_csv('./cost_matrix/cost_matrix.txt', header=None)
    cost_array = np.array(df_cost_M)
    total_cost = np.sum(cm * cost_array)
    return total_cost

def minmax(array, Min, Max):
    row, col = array.shape
    output = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            scale = (array[i, j] - array[i, :].min()) * (Max - Min) / (array[i, :].max() - array[i, :].min()) + Min
            output[i, j] = scale
    return output
