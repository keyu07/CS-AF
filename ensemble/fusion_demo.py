import numpy as np
import pandas as pd
import random
from util import multi_class_F1score, minmax, sens_spec, total_cost
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

y_df = pd.read_csv('./SF/dist_1/SF_test/y_true.txt', header=None)
y_true = np.array(y_df)
lens=len(y_df)


def maxvote(pick_list, cost_file):
    y_pred = np.zeros((lens, 1))
    voting_table = np.zeros((lens, len(pick_list))).astype(int)
    model_id = 0
    for i in pick_list:
        x_df = pd.read_csv('{}'.format(sf_test_list[i]), header=None)
        x_vec = np.array(x_df)
        predict = np.argmax(x_vec, 1)
        voting_table[:, model_id] = predict
        model_id = model_id + 1

    for row in range(lens):
        count = np.bincount(voting_table[row, :])
        y_pred[row] = np.argmax(count)
        
    # Compute the over all accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate CM (normalized) and compute total cost
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cost = total_cost(cm, cost_file)

    return accuracy, cost

def ave(pick_list, cost_file):
    # Define the target model when comes a query
    final_x_vec = np.zeros((lens,8))
    for i in pick_list:
        x_df = pd.read_csv('{}'.format(sf_test_list[i]), header=None)
        x_vec = np.array(x_df)
        final_x_vec = x_vec + final_x_vec
    
    y_pred = np.argmax(final_x_vec, 1)
    # Compute the over all accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)


    # Generate CM (normalized) and compute total cost
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cost = total_cost(cm, cost_file)
    
    return accuracy, cost

def cs_af(cost_file_F1, cost_file_test, pick_list, alpha=0.5):
    F1_score = np.array([])
    final_softlabel = np.zeros((lens, 8))
    for i in pick_list:
        val_df = pd.read_csv('{}'.format(sf_val_list[i]), header=None)
        F1 = multi_class_F1score(val_df, cost_file_F1, mode='cs-af')
        F1_score = np.append(F1_score, F1)
    F1_score = np.reshape(F1_score, (1, len(pick_list)))
    F1_score = minmax(F1_score, 0.1, 0.5)
    for i in pick_list:
        x_df = pd.read_csv('{}'.format(sf_test_list[i]), header=None)
        x_vec = np.array(x_df)
        confidence_score = (1 - alpha) * x_vec.max(1) + alpha * F1_score[pick_list.index(i)]
        confidence_score = np.reshape(confidence_score, (lens, 1))
        final_softlabel = final_softlabel + (x_vec * confidence_score)
        
    y_pred = np.argmax(final_softlabel, 1)
    # Compute the over all accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute Sensitivity and Specificity
    sens, spec = sens_spec(cm)

    # Generate CM (normalized) and compute total cost
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cost = total_cost(cm, cost_file_test)

    return accuracy, cost

def af(cost_file, pick_list, alpha=0.5):
    F1_score = np.array([])
    final_softlabel = np.zeros((lens, 8))
    for i in pick_list:
        val_df = pd.read_csv('{}'.format(sf_val_list[i]), header=None)
        F1 = multi_class_F1score(val_df, cost_file, mode='af')
        F1_score = np.append(F1_score, F1)
    F1_score = np.reshape(F1_score, (1, len(pick_list)))
    F1_score = minmax(F1_score, 0.1, 0.5)
    for i in pick_list:
        x_df = pd.read_csv('{}'.format(sf_test_list[i]), header=None)
        x_vec = np.array(x_df)
        confidence_score = (1 - alpha) * x_vec.max(1) + alpha * F1_score[pick_list.index(i)]
        confidence_score = np.reshape(confidence_score, (lens, 1))
        final_softlabel = final_softlabel + (x_vec * confidence_score)
        
    y_pred = np.argmax(final_softlabel, 1)
    # Compute the over all accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Generate CM (normalized) and compute total cost
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cost = total_cost(cm, cost_file)

    return accuracy, cost

picks = [8,16,24,32,40,48]
attempt = 3

# Over all accuracy with different cost matrices
max_all_acc = np.zeros((attempt, len(picks)))
ave_all_acc = np.zeros((attempt, len(picks)))
af_all_acc = np.zeros((attempt, len(picks)))
csaf_all_acc1 = np.zeros((attempt, len(picks)))
csaf_all_acc2 = np.zeros((attempt, len(picks)))

# Toal cost with different cost matrices
max_costA = np.zeros((attempt, len(picks)))
ave_costA = np.zeros((attempt, len(picks)))
af_costA = np.zeros((attempt, len(picks)))
csaf_costA = np.zeros((attempt, len(picks)))

max_costB = np.zeros((attempt, len(picks)))
ave_costB = np.zeros((attempt, len(picks)))
af_costB = np.zeros((attempt, len(picks)))
csaf_costB = np.zeros((attempt, len(picks)))

models = ['dpn', 'efficientb7', 'incepresv2', 
          'inceptionv3', 'inceptionv4', 'nasnet', 
          'pnasnet', 'resnet152', 'resnext101', 
          'se_resnext101', 'senet154', 'xception']


sf_test_list = []
sf_val_list = []
for i in range(48):
    for model in models:
        sf_val_list.append('./SF/dist_{}/SF_val/SF_{}.txt'.format(int(i/12)+1, model))
        sf_test_list.append('./SF/dist_{}/SF_test/SF_{}.txt'.format(int(i/12)+1, model))

for trial in range(attempt):

     print('############################# Running - {}/{} #############################'.format(trial+1, attempt))

     cost_file1 = 'cost_matrix_A'
     cost_file2 = 'cost_matrix_B'

     for pick in picks:
         
        lis = random.sample(range(48), pick)
        
        # results of Max ensemble
        max_acc, costA = maxvote(lis, cost_file1)
        _, costB= maxvote(lis, cost_file2)
        
        max_all_acc[trial, picks.index(pick)] = max_acc
        max_costA[trial, picks.index(pick)] = costA
        max_costB[trial, picks.index(pick)] = costB
        
        # results of average ensemble
        ave_acc, costA = ave(lis, cost_file1)
        _, costB = ave(lis, cost_file2)
        
        ave_all_acc[trial, picks.index(pick)] = ave_acc
        ave_costA[trial, picks.index(pick)] = costA
        ave_costB[trial, picks.index(pick)] = costB
        
        # results of AF
        af_acc, costA = af(cost_file1, lis)
        _, costB = af(cost_file2, lis)
        
        af_all_acc[trial, picks.index(pick)] = af_acc
        af_costA[trial, picks.index(pick)] = costA
        af_costB[trial, picks.index(pick)] = costB
        
        # results of CS-AF
        over_all_accA, costA= cs_af(cost_file1, cost_file1, lis)
        over_all_accB, costB = cs_af(cost_file2, cost_file2, lis)
        
        csaf_all_acc1[trial, picks.index(pick)] = over_all_accA
        csaf_all_acc2[trial, picks.index(pick)] = over_all_accB
        csaf_costA[trial, picks.index(pick)] = costA
        csaf_costB[trial, picks.index(pick)] = costB


# Plot the over all accuracy 
max_acc = np.mean(max_all_acc, axis=0)
ave_acc = np.mean(ave_all_acc, axis=0)
af_acc = np.mean(af_all_acc,  axis=0)
csaf1_acc = np.mean(csaf_all_acc1,  axis=0)
csaf2_acc = np.mean(csaf_all_acc2,  axis=0)
plt.figure(figsize=(20,6))
plt.subplot(131)
plt.plot(picks, max_acc, color='b', marker='s',linestyle='-.')
plt.plot(picks, ave_acc, color='darkorange', marker='^', linestyle='-')
plt.plot(picks, af_acc, color='darkorchid', marker='o', linestyle='--')
plt.plot(picks, csaf1_acc, 'g:x')
plt.plot(picks, csaf2_acc, 'r-+')
plt.legend(('Max Voting', 'Average ','AF', 'CS-AF (Cost Matrix A)', 'CS-AF (Cost Matrix B)',), fontsize=14, loc='best', ncol=1)
plt.xlabel('# of models', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.xticks(picks, fontsize=14)
plt.yticks(fontsize=14)




# Plot the total cost with cost matrix A
max_costA = np.mean(max_costA, axis=0)
ave_costA = np.mean(ave_costA, axis=0)
af_costA = np.mean(af_costA, axis=0)
csaf_costA = np.mean(csaf_costA, axis=0)
plt.subplot(132)
plt.plot(picks, max_costA, color='b', marker='s',linestyle='-.')
plt.plot(picks, ave_costA, color='darkorange', marker='^', linestyle='-')
plt.plot(picks, af_costA, color='darkorchid', marker='o', linestyle='--')
plt.plot(picks, csaf_costA, 'g:x')
plt.legend(('Max Voting', 'Average ','AF', 'CS-AF (Cost Matrix A)'), fontsize=14, loc='best', ncol=1)
plt.xlabel('# of models', fontsize=15)
plt.ylabel('Total Cost', fontsize=15)
plt.title('Evaluate with Cost Matrix A')
plt.xticks(picks, fontsize=14)
plt.yticks(fontsize=14)




# Plot the total cost with cost matrix B
max_costB = np.mean(max_costB, axis=0)
ave_costB = np.mean(ave_costB, axis=0)
af_costB = np.mean(af_costB, axis=0)
csaf_costB = np.mean(csaf_costB, axis=0)
plt.subplot(133)
plt.plot(picks, max_costB, color='b', marker='s',linestyle='-.')
plt.plot(picks, ave_costB, color='darkorange', marker='^', linestyle='-')
plt.plot(picks, af_costB, color='darkorchid', marker='o', linestyle='--')
plt.plot(picks, csaf_costB, 'r-+')
plt.legend(('Max Voting', 'Average ','AF', 'CS-AF (Cost Matrix B)'), fontsize=14, loc='best', ncol=1)
plt.xlabel('# of models', fontsize=15)
plt.ylabel('Total Cost', fontsize=15)
plt.title('Evaluate with Cost Matrix B')
plt.xticks(picks, fontsize=14)
plt.yticks(fontsize=14)
