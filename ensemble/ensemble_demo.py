import numpy as np
import pandas as pd
from util import multi_class_F1score, total_cost, minmax, compute_acc

models = ['pnasnet','nasnet', 'resnext101', 'senet154', 'dpn',
          'xception', 'inceptionv4', 'incepresv2', 'se_resxt101', 'resnet152']

def max_voting():
    voting_table = np.zeros((3776, len(models))).astype(int)
    final_predict = np.zeros((3776,1))
    for model in models:
        x_df = pd.read_csv('./SF/SF_{}_full.txt'.format(model), header=None)
        x_vector = np.array(x_df)
        y_predict = np.argmax(x_vector, 1)
        voting_table[:, models.index(model)] = y_predict

    for row in range(3776):
        count = np.bincount(voting_table[row, :])
        final_predict[row] = np.argmax(count)

    accuracy = compute_acc(final_predict)
    cost = total_cost(final_predict)
    return accuracy, cost

def average_ensemble():
    final_softlabel = np.zeros((3776, 8))
    for model in models:
        x_df = pd.read_csv('./SF/SF_{}_full.txt'.format(model), header=None)
        x_vector = np.array(x_df)
        final_softlabel = x_vector + final_softlabel

    final_predict = np.argmax(final_softlabel, 1)
    accuracy = compute_acc(final_predict)
    cost = total_cost(final_predict)
    return accuracy, cost

def cs_af(alpha=0.7):
    F1_score = np.array([])
    final_softlabel = np.zeros((3776, 8))
    for model in models:
        val_df = pd.read_csv('./val_cm/cm_{}_full.txt'.format(model), header=None)
        F1 = multi_class_F1score('cost', val_df)
        F1_score = np.append(F1_score, F1)
    F1_score = np.reshape(F1_score, (1, 10))
    F1_score = minmax(F1_score, 0.1, 0.5)
    F1_score = np.reshape(F1_score, (10, 1))

    for model in models:
        x_df = pd.read_csv('./SF/SF_{}_full.txt'.format(model), header=None)
        x_vec = np.array(x_df)
        confidence_score = (1 - alpha) * x_vec.max(1) + alpha * F1_score[models.index(model)]
        confidence_score = np.reshape(confidence_score, (3776, 1))
        final_softlabel = final_softlabel + (x_vec * confidence_score)

    final_predict = np.argmax(final_softlabel, 1)
    accuracy = compute_acc(final_predict)
    cost = total_cost(final_predict)
    return accuracy, cost

def af(alpha=0.7):
    F1_score = np.array([])
    final_softlabel = np.zeros((3776, 8))
    for model in models:
        val_df = pd.read_csv('./val_cm/cm_{}_full.txt'.format(model), header=None)
        F1 = multi_class_F1score('nocost', val_df)
        F1_score = np.append(F1_score, F1)
    F1_score = np.reshape(F1_score, (1, 10))
    F1_score = minmax(F1_score, 0.1, 0.5)
    F1_score = np.reshape(F1_score, (10, 1))

    for model in models:
        x_df = pd.read_csv('./SF/SF_{}_full.txt'.format(model), header=None)
        x_vec = np.array(x_df)
        confidence_score = (1 - alpha) * x_vec.max(1) + alpha * F1_score[models.index(model)]
        confidence_score = np.reshape(confidence_score, (3776, 1))
        final_softlabel = final_softlabel + (x_vec * confidence_score)

    final_predict = np.argmax(final_softlabel, 1)
    accuracy = compute_acc(final_predict)
    cost = total_cost(final_predict)
    return accuracy, cost

max_acc, max_cost = max_voting()
ave_acc, ave_cost = average_ensemble()
cs_af_acc, cs_af_cost = cs_af()
af_acc, af_cost = af()
