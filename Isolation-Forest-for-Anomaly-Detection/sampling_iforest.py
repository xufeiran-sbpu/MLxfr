from sklearn.datasets import load_breast_cancer
import pandas as pd
import pickle
import math
import random

import numpy as np
import itertools
import bisect
import math
import pandas as pd
import warnings
import random
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import sys
import time
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, \
    confusion_matrix, f1_score, average_precision_score
from iforest import IsolationTreeEnsemble, find_TPR_threshold
from scipy.fftpack import fft, ifft
from matplotlib.pylab import mpl
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics
from sklearn.metrics import mean_squared_error

#读取数据集
cancer = load_breast_cancer()
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['diagnosis'] = cancer.target
df.loc[df.diagnosis == 0, 'diagnosis'] = -1
df.loc[df.diagnosis == 1, 'diagnosis'] = 0
df.loc[df.diagnosis == -1, 'diagnosis'] = 1
df.to_csv("cancer.csv", index=False)


#孤立森林算法载入
def plot_anomalies(X, y, sample_size=256, n_trees=100, desired_TPR=None, percentile=None, normal_ymax=None, bins=20):
    N = len(X)
    it = IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    print(f"fit time {fit_time:3.2f}s")
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    print(f"score time {score_time:3.2f}s")
    if desired_TPR is not None:
        threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
    else:
        threshold = np.percentile(scores, percentile)
    y_pred = it.predict_from_anomaly_scores(scores, threshold=threshold)
    confusion = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    normal = scores[y == 0]
    anomalies = scores[y == 1]
    F1 = f1_score(y, y_pred)
    PR = average_precision_score(y, scores)
    print(f"Proportion anomalies/normal = {len(anomalies)}/{len(normal)} = {(len(anomalies) / len(normal)) * 100:.1f}%")
    print(f"F1 score {F1:.4f}, avg PR {PR:.4f}")
    fig, axes = plt.subplots(2, 1, sharex=True)
    counts0, binlocs0, _ = axes[0].hist(normal, color='#c7e9b4', bins=bins)
    counts1, binlocs1, _ = axes[1].hist(anomalies, color='#fee090', bins=bins)
    axes[1].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Normal sample count")
    axes[1].set_ylabel("Anomalous sample count")
    axes[0].plot([threshold, threshold], [0, max(counts0)], '--', color='grey')
    axes[1].plot([threshold, threshold], [0, max(counts1)], '--', color='grey')
    text_xr = 0.97 * axes[0].get_xlim()[1]
    axes[0].text(text_xr, .85 * max(counts0), f"N {N}, {n_trees} trees", horizontalalignment='right')
    axes[0].text(text_xr, .75 * max(counts0), f"F1 score {F1:.4f}, avg PR {PR:.4f}", horizontalalignment='right')
    axes[0].text(text_xr, .65 * max(counts0), f"TPR {TPR:.4f}, FPR {FPR:.4f}", horizontalalignment='right')
    axes[0].text(threshold + .005, .20 * max(counts0), f"score threshold {threshold:.3f}")
    axes[0].text(threshold + .005, .10 * max(counts0), f"True anomaly rate {len(anomalies) / len(normal):.4f}")
    if normal_ymax is not None:
        axes[0].set_ylim(0, normal_ymax)
    plt.tight_layout()
    plt.savefig(f"{datafile.split('.')[0]}-{n_trees}-{int(desired_TPR * 100)}.svg",
                bbox_inches='tight',
                pad_inches=0)

    return F1
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
def ApproShap_shou(ListOfSampling, characteristicF, playerList, counter):
    sha = [0 for i in range(len(playerList))]
    shaOld = [0 for i in range(len(playerList))]
    errors = []
    k = 0
    for ob in ListOfSampling:
        shaNew = [0 for i in range(len(playerList))]
        for i in ob:  # ob is sequence of player id from 0 to N
            pre_i_ob = calculatePre_i(ob, i)  # calculate pre^i(ob)
            x_ob = calculateX(pre_i_ob, i, characteristicF[k])
            sha[i] = sha[i] + x_ob
        shaNew = sha.copy()

        k = k + 1
        print("k", k)
        tempError = 0
        if k == 1:
            tempError = mean_squared_error(shaNew, shaOld)
            print("first:", tempError)
        else:
            t1 = [x / (k - 1) for x in shaNew]
            t2 = [x / (k - 1) for x in shaOld]
            tempError = mean_squared_error(t1, t2)
            print("第2次上,t1,t1:,error", t1, t2, tempError)
        errors.append(tempError)
        shaOld = shaNew.copy()
    sha = [x / np.shape(ListOfSampling)[0] for x in sha]
    print("errors:", errors)
    return sha, errors
def characteristicFunction(ob, characteristicF):
    result = 0
    # print("ob",ob)
    if ob == None:
        return result
    else:
        result = characteristicF[len(ob) - 1]
    return result
def calculateX(ob, i, characteristicF):
    new_ob = ob.copy()
    new_ob.append(i)
    # print("new_ob,",new_ob)
    # print("ob",ob)
    result = characteristicFunction(new_ob, characteristicF) - characteristicFunction(ob, characteristicF)
    return result
def calculatePre_i(ob, i):
    result = []
    for k in range(0, len(ob)):
        if ob[k] != i:
            result.append(ob[k])
        else:
            break

    return result
def permutation(list, count):  # probability: 1/n!, count: sampling size
    sampling = []
    for k in range(0, count):
        templist = list.copy()
        random.shuffle(list)
        sampling.append(templist)
    return sampling
def randomChose(probability, listPlayer):
    np.random.seed(0)
    p = np.array()
    for k in list:
        np.append(p, probability, axis=0)
    index = np.random.choice(listPlayer, p=p.ravel())
    return listPlayer[index]
def ApproShap(ListOfSampling, characteristicF, playerList, counter):
    sha = [0 for i in range(len(playerList))]
    for ob in ListOfSampling:
        k = 0
        for i in ob:  # ob is sequence of player id from 0 to N
            pre_i_ob = calculatePre_i(ob, i)  # calculate pre^i(ob)
            #             print("k",k)
            x_ob = calculateX(pre_i_ob, i, characteristicF[k])
            sha[i] = sha[i] + x_ob
        k = k + 1

    sha = [x / np.shape(ListOfSampling)[0] for x in sha]
    return sha
def pr_anomalies(X, y, sample_size=256, n_trees=100, desired_TPR=None, percentile=None, normal_ymax=None, bins=20):
    N = len(X)
    it = IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)
    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    #     print(y, scores, desired_TPR)
    if desired_TPR is not None:
        try:
            threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        # print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
        except:
            print(y, scores, desired_TPR)
            print(type(y), type(scores), type(desired_TPR))
    else:
        threshold = np.percentile(scores, percentile)
    y_pred = it.predict_from_anomaly_scores(scores, threshold=threshold)
    confusion = confusion_matrix(y, y_pred)

    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    normal = scores[y == 0]

    anomalies = scores[y == 1]
    F1 = f1_score(y, y_pred)
    PR = average_precision_score(y, scores)

    return PR

if __name__ == '__main__':  # dask seems to need this

    datafile = 'cancer.csv'
    #
    targetcol = 'diagnosis'
    sample_size = int(5)
    n_trees = int(300)
    desired_TPR = int(80)
    desired_TPR /= 100.0

    X, y = df.drop(targetcol, axis=1), df[targetcol]
    plot_anomalies(X, y, sample_size=sample_size, n_trees=n_trees, desired_TPR=desired_TPR, bins=15)
    X_ = np.array(X)
    print("!!!!!!!!!!")
    playerList = []
    for i in range(30):
        playerList.append(i)
    counter = 50  # sampling size
    # characteristicF =[np.zeros(len(playerList))]*counter
    # initial table for characteristic function of coalition in permutation
    samplist = []
    start_time = time.time()
    characteristicF = []
    P = []
    ob = playerList
    rep = {}
    listOfSampling = permutation(ob, counter)
    # print(listOfSampling)
    for i in range(counter):
        for j in range(30):
            #         if j == 0:
            #             data_samp_x = getgroup_level(X_,listOfSampling[i][0])
            #             size_zeros = 29
            #         else:
            #             data_samp_x = getgroup_level(X_,listOfSampling[i][:j])
            #             size_zeros =30 - data_samp_x.shape[1]
            #         data_samp = np.c_[data_samp_x,np.zeros((569,size_zeros))]
            # #         print(np.shape(data_samp))
            X_copy = X_.copy()
            X_copy[:, j + 1:30] = np.zeros((569, 29 - j))

            pr = pr_anomalies(X_copy, y, n_trees=n_trees, desired_TPR=0.8)
            P.append(pr)
        # print("characteristic function:",j,P,X_copy,)
        rep[i] = P
        # print(rep)
    RP = np.array(P)
    characteristicF = RP.reshape((counter, 30))
    #     print(i)

    sha2, ERR = ApproShap_shou(listOfSampling, characteristicF, playerList, counter)
    plt.figure(figsize=(12, 9))
    N_ = np.linspace(1,counter,counter)
    plt.plot(N_[1:], ERR[1:], 'r-')
    plt.show()
    plt.savefig('shoulian_80.png')

    #

    print(counter)
    start_time = time.time()
    sha = ApproShap(listOfSampling, characteristicF, playerList, counter)
    S_ = sum(abs(np.array(sha)))
    Shap_samp = abs(np.array(sha)) / S_
    stop_time = time.time()
    time_cacul = stop_time - start_time
    print(f"time {time_cacul:3.2f}s")
    print(Shap_samp)

    #

    playerlist = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', \
                  'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
                  'mean fractal dimension', 'radius error', \
                  'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', \
                  'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
                  'worst radius', \
                  'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
                  'worst concavity', \
                  'worst concave points', 'worst symmetry', 'worst fractal dimension']
    Shapley = Shap_samp
    df = pd.DataFrame({'eventlist': playerlist, 'shapleys': Shapley})
    df = df.sort_values(by=['shapleys'], axis=0, ascending=[True])
    # shapleys = [0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404, 0.0027905302007381404]
    # plt.bar(shapleys,range(len(eventlist)), color='lightsteelblue')
    plt.figure(figsize=(12, 9))
    plt.bar(x=0, bottom=np.arange(len(df['shapleys'])), height=0.9, width=df['shapleys'], orientation="horizontal",
            color='lightsteelblue')
    plt.plot(df['shapleys'], range(len(df['eventlist'])), marker='o', color='coral')  # coral
    plt.yticks(range(len(df['eventlist'])), df['eventlist'])
    plt.show()
    plt.savefig('Shap_samp_1000.png')
    print("@@#@#@#@#@#@#@#")



# def ApproShap_shou(ListOfSampling, characteristicF, playerList, counter):
#     sha = [0 for i in range(len(playerList))]
#     shaOld = [0 for i in range(len(playerList))]
#     errors = []
#     for ob in ListOfSampling:
#         k = 0
#         shaNew = [0 for i in range(len(playerList))]
#         for i in ob:  # ob is sequence of player id from 0 to N
#             pre_i_ob = calculatePre_i(ob, i)  # calculate pre^i(ob)
#             #             print("k",k)
#             x_ob = calculateX(pre_i_ob, i, characteristicF[k])
#             sha[i] = sha[i] + x_ob
#             shaNew[i] = x_ob
#             print("::", shaNew)
#         k = k + 1
#
#         tempError = 0
#         for i in range(0, len(shaNew)):
#             tempError = tempError + np.sqrt(shaNew[i] - shaOld[i])
#         print("!!!!", tempError)
#         errors.append(tempError)
#         shaOld = shaNew.copy()
#     sha = [x / np.shape(ListOfSampling)[0] for x in sha]
#     return errors
# X, y = df.drop(targetcol, axis=1), df[targetcol]
# X_ = np.array(X)
# playerList = []
# for i in range(30):
#     playerList.append(i)
# counter = 15  # sampling size
# # characteristicF =[np.zeros(len(playerList))]*counter
# # initial table for characteristic function of coalition in permutation
# samplist = []
# start_time = time.time()
# characteristicF = []
# P = []
# ob = playerList
# rep = {}
# listOfSampling = permutation(ob, counter)
# # print(listOfSampling)
# for i in range(counter):
#     for j in range(30):
#         #         if j == 0:
#         #             data_samp_x = getgroup_level(X_,listOfSampling[i][0])
#         #             size_zeros = 29
#         #         else:
#         #             data_samp_x = getgroup_level(X_,listOfSampling[i][:j])
#         #             size_zeros =30 - data_samp_x.shape[1]
#         #         data_samp = np.c_[data_samp_x,np.zeros((569,size_zeros))]
#         # #         print(np.shape(data_samp))
#         X_copy = X_.copy()
#         X_copy[:, j + 1:30] = np.zeros((569, 29 - j))
#
#         pr = pr_anomalies(X_copy, y, n_trees=n_trees, desired_TPR=0.8)
#         P.append(pr)
#     # print("characteristic function:",j,P,X_copy,)
#     rep[i] = P
#     # print(rep)
# RP = np.array(P)
# characteristicF = RP.reshape((counter, 30))
# #     print(i)
#
# sha = ApproShap(listOfSampling, characteristicF, playerList, counter)
# stop_time = time.time()
# time_cacul = stop_time - start_time
# print(f"time {time_cacul:3.2f}s")
# print("shapley", sha)

#

# 收敛图

#



# mse = mean_squared_error(A, B)



