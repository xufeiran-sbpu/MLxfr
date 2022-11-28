import math
import random
import numpy as np
from sklearn.metrics import mean_squared_error

import numpy as np
import itertools
import bisect
import math
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import sys
import time
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, \
    confusion_matrix, f1_score, average_precision_score
from iforest import IsolationTreeEnsemble, find_TPR_threshold
from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics

from sklearn.datasets import load_breast_cancer
import datetime

def ApproShap(ListOfSampling,characteristicF,playerList,counter):
    sha=[0 for i in range(len(playerList))]
    for ob in ListOfSampling:
        k=0
        for i in ob:# ob is sequence of player id from 0 to N
            pre_i_ob=calculatePre_i(ob,i)#calculate pre^i(ob)
            print("k",k)
            x_ob=calculateX(pre_i_ob,i,characteristicF[k])
            sha[i]=sha[i]+x_ob
        k=k+1

    sha=[x/np.shape(ListOfSampling)[0] for x in sha]
    return sha

def randomChose(probability,listPlayer):
    np.random.seed(0)
    p = np.array()
    for k in list:
        np.append(p,probability,axis=0)
    index = np.random.choice(listPlayer, p=p.ravel())
    return listPlayer[index]

def permutation(list,count): #probability: 1/n!, count: sampling size
    sampling=[]
    for k in range(0,count):
        templist = list.copy()
        random.shuffle(list)
        sampling.append(templist)
    return sampling



def calculatePre_i(ob,i):
    result=[]
    for k in range(0,len(ob)):
        if ob[k]!=i:
            result.append(ob[k])
        else :
            break

    return result

def calculateX(ob,i,characteristicF):
    new_ob=ob.copy()
    new_ob.append(i)
   # print("new_ob,",new_ob)
   # print("ob",ob)
    result = characteristicFunction(new_ob,characteristicF)-characteristicFunction(ob,characteristicF)
    return result


def characteristicFunction(ob,characteristicF):
    result=0
    #print("ob",ob)
    if ob==None:
        return result
    else:
        result=characteristicF[len(ob)-1]
    return result

def pr_anomalies(it,X, y,threshold, sample_size=256, n_trees = 100, desired_TPR=None, percentile = None, normal_ymax=None, bins=20):
    N = len(X)

    score, y_pred = it.predictSingleInstance(X, threshold=threshold)

    return score

def sampling_explianer(X_,y):
    instanceSize = X_.shape[0]
    ResultOfSampling=[]
    it = IsolationTreeEnsemble(sample_size=256, n_trees=100)
    fit_start = time.time()
    it.fit(X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    score_start = time.time()
    scores = it.anomaly_score(X)
    score_stop = time.time()
    score_time = score_stop - score_start
    threshold=0.5
    #     print(y, scores, desired_TPR)

    if desired_TPR is not None:
        try:
            threshold, FPR = find_TPR_threshold(y, scores, desired_TPR)
        # print(f"Computed {desired_TPR:.4f} TPR threshold {threshold:.4f} with FPR {FPR:.4f}")
        except:
            print(y, scores, desired_TPR)
            print(type(y), type(scores), type(desired_TPR))
    else:
        threshold = np.percentile(scores, percentile=None)



    X_copyup = X_.copy()
    yup=y

    for bigi in range(0,instanceSize):
        X_ = X_copyup[bigi, :]
        y = yup[bigi]
        X_ = X_.reshape((1, 30))
        sample_size = X_.shape[0]
        print(bigi)
        n_trees = int(300)
        playerList = []
        for i in range(30):
            playerList.append(i)
        counter = 10  # sampling size
        # characteristicF =[np.zeros(len(playerList))]*counter
        # initial table for characteristic function of coalition in permutation
        samplist = []
        characteristicF = []
        P = []
        ob = playerList
        rep = {}
        listOfSampling = permutation(ob, counter)
        # print(listOfSampling)

        for i in range(counter):
            for j in range(30):
                X_copy = X_.copy()
                X_copy[:, j + 1:30] = np.zeros((sample_size, 29 - j))
                pr = pr_anomalies(it, X_copy, y, threshold, n_trees=n_trees, desired_TPR=0.8)
                P.append(pr)
            # print("characteristic function:",j,P,X_copy,)
            rep[i] = P
            # print(rep)
        RP = np.array(P)
        characteristicF = RP.reshape((counter, 30))
        #     print(i)

        sha2, ERR = ApproShap_shou(listOfSampling, characteristicF, playerList, counter)
        if(bigi==0):
            ResultOfSampling=np.array(sha2)
        else:
            ResultOfSampling=np.append(ResultOfSampling,np.array(sha2),axis=0)

    ResultOfSampling=ResultOfSampling.reshape(instanceSize,30)
    str = '/Users/jes/Desktop/学/testresult/samplingresult' + datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S") + '.txt'

    np.savetxt(str, ResultOfSampling)



    return ResultOfSampling


def ApproShap_shou(ListOfSampling,characteristicF,playerList,counter):
    sha=[0 for i in range(len(playerList))]
    shaOld=[0 for i in range(len(playerList))]
    errors=[]
    k=0
    for ob in ListOfSampling:
        shaNew=[0 for i in range(len(playerList))]
        for i in ob:# ob is sequence of player id from 0 to N
            pre_i_ob=calculatePre_i(ob,i)#calculate pre^i(ob)
            x_ob=calculateX(pre_i_ob,i,characteristicF[k])
            sha[i]=sha[i]+x_ob
        shaNew=sha.copy()

        k=k+1
       # print("k",k)
        tempError=0
        for i in range(0,len(shaNew)):
             tempError= tempError+ np.sqrt(np.abs(shaNew[i]-shaOld[i]))
        errors.append(tempError)
        shaOld=shaNew.copy()
    sha=[x/np.shape(ListOfSampling)[0] for x in sha]
   # print("errors:",errors)
    return sha,errors



if __name__ == '__main__':
    print("test")
    cancer = load_breast_cancer()
    df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = cancer.target
    df.loc[df.diagnosis==0,'diagnosis'] = -1
    df.loc[df.diagnosis==1,'diagnosis'] = 0
    df.loc[df.diagnosis==-1,'diagnosis'] = 1
    df.to_csv("cancer.csv", index=False)

    # load JS visualization code to notebook

    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    suss= np.loadtxt('/Users/jes/Desktop/学/testresult/samplingresult2020-12-09-01-54-40.txt')
    print("loading",suss)

    datafile = 'cancer.csv'

    targetcol = 'diagnosis'
    sample_size = int(5)
    n_trees = int(300)
    desired_TPR = int(80)
    desired_TPR /= 100.0

    X, y = df.drop(targetcol, axis=1), df[targetcol]
    X_ = np.array(X)

    sampling_explianer(X_,y)










