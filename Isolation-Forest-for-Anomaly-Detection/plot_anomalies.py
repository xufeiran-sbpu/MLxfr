import numpy as np
import itertools
import bisect
import math
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, \
    confusion_matrix, f1_score, average_precision_score
import matplotlib.pyplot as plt
import sys
import time

from iforest import IsolationTreeEnsemble, find_TPR_threshold

# def add_noise(df, n_noise):
#     for i in range(n_noise):
#         df[f'noise_{i}'] = np.random.normal(-2,2,len(df))


def plot_anomalies(X, y, sample_size=256, n_trees = 100, desired_TPR=None, percentile = None, normal_ymax=None, bins=20):
    N = len(X)
    # print("____________",X)
    it = IsolationTreeEnsemble(sample_size=sample_size, n_trees=n_trees)

    fit_start = time.time()
    it.fit(X)
    # print("************",X)
    fit_stop = time.time()
    fit_time = fit_stop - fit_start
    print(f"fit time {fit_time:3.2f}s")

    score_start = time.time()
    scores = it.anomaly_score(X)
    # print("这里是scores:",scores)
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
    # print(confusion)

    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    normal = scores[y==0]
    anomalies = scores[y==1]
    F1 = f1_score(y, y_pred)
    PR = average_precision_score(y, scores)
    print(f"Proportion anomalies/normal = {len(anomalies)}/{len(normal)} = {(len(anomalies)/len(normal))*100:.1f}%")
    print(f"F1 score {F1:.4f}, avg PR {PR:.4f}")

    fig, axes = plt.subplots(2, 1, sharex=True)


    counts0, binlocs0, _ = axes[0].hist(normal, color='#c7e9b4', bins=bins)
    counts1, binlocs1, _ = axes[1].hist(anomalies, color='#fee090', bins=bins)
    # print("COUNTS0:", counts0)
    # print("BINLOCS0:",binlocs0)
    # print("COUNTS1:", counts1)
    # print("BINLOCS1:",binlocs1)
    axes[1].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Normal sample count")
    axes[1].set_ylabel("Anomalous sample count")
    axes[0].plot([threshold,threshold],[0,max(counts0)], '--', color='grey')
    axes[1].plot([threshold,threshold],[0,max(counts1)], '--', color='grey')
    text_xr = 0.97 * axes[0].get_xlim()[1]
    axes[0].text(text_xr, .85 * max(counts0), f"N {N}, {n_trees} trees", horizontalalignment='right')
    axes[0].text(text_xr, .75 * max(counts0), f"F1 score {F1:.4f}, avg PR {PR:.4f}", horizontalalignment='right')
    axes[0].text(text_xr, .65 * max(counts0), f"TPR {TPR:.4f}, FPR {FPR:.4f}", horizontalalignment='right')
    axes[0].text(threshold+.005, .20 * max(counts0), f"score threshold {threshold:.3f}")
    axes[0].text(threshold+.005, .10 * max(counts0), f"True anomaly rate {len(anomalies) / len(normal):.4f}")
    if normal_ymax is not None:
        axes[0].set_ylim(0, normal_ymax)
    plt.tight_layout()
    plt.savefig(f"{datafile.split('.')[0]}-{n_trees}-{int(desired_TPR*100)}.svg",
                bbox_inches='tight',
                pad_inches=0)
    # plt.show()
    return F1
def calculatShapley(cFunction,coalition,nPlayer):
    coalition=list(coalition)
    for i in range(0,len(coalition)):
        coalition[i]=list(coalition[i])


    print("start calculate shapley:")
    shapley_values = []
    for i in range(len(nPlayer)):
        shapley = 0
        for j in coalition:
            if i not in j:
                j=list(j)
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = coalition.index(j)
                k = coalition.index(Cui)
                temp = float(float(cFunction[k]) - float(cFunction[l])) *\
                           float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = coalition.index(Cui)
        temp = float(cFunction[k]) * float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
        shapley += temp

        shapley_values.append(shapley)

    return (shapley_values)
def get_subsets(whole_set):
    output = [[]]
    for i in whole_set:
        output.extend([subset + [i] for subset in output])
    subsets_list = [set(x) for x in output[1:-1]]
    return subsets_list

def getcoaltionlist():
    coalition=[]
    for i in range(1, 3):
        for p in itertools.combinations((0, 1, 2), i):
            coalition.append(p)
            # if i==2:
            #     print(p)
    return coalition

def data_split(split):
    a = np.zeros_like(split[:, 0])
    for i in range(9):
        data_1 = np.concatenate((a,split[:,i+1]),axis=0)
        a = data_1
    return a
if __name__ == '__main__': # dask seems to need this
    # launch with "python plot_anomalies.py http.csv attack 20000 256 100 99"
    # or, "python plot_anomalies.py cancer.csv diagnosis all 5 1000 80

    # datafile = sys.argv[1]
    # targetcol = sys.argv[2]
    # sample_size = int(sys.argv[4])
    # n_trees = int(sys.argv[5])
    # desired_TPR = int(sys.argv[6])
    datafile = 'cancer.csv'

    targetcol = 'diagnosis'
    sample_size = int(5)
    n_trees = int(300)
    desired_TPR = int(80)
    desired_TPR /= 100.0

    df = pd.read_csv(datafile)
    df_list=np.array(df)
    print(np.shape(df_list))
    tm = df_list[:,0:30]
    tm1=df_list[:,0:10].reshape(-1,10)
    tm2=df_list[:,10:20].reshape(-1,10)
    tm3=df_list[:,20:30].reshape(-1,10)

    # a1=data_split(tm1).reshape(-1,1)
    # a2=data_split(tm2).reshape(-1,1)
    # a3=data_split(tm3).reshape(-1,1)
    # layer_1 = np.concatenate((tm1,tm2,tm3),axis=1)
    layer1_F1 = plot_anomalies(tm1, df_list[:,-1], sample_size=sample_size, n_trees=n_trees, desired_TPR=desired_TPR, bins=15)
    layer2_F1 = plot_anomalies(tm2, df_list[:,-1], sample_size=sample_size, n_trees=n_trees, desired_TPR=desired_TPR, bins=15)
    layer3_F1 = plot_anomalies(tm2, df_list[:,-1], sample_size=sample_size, n_trees=n_trees, desired_TPR=desired_TPR, bins=15)
    S1= np.array([layer1_F1,layer2_F1,layer3_F1])
    print(S1)
    # print(df_list[:,-1])
    subset = get_subsets([0,1,2])
    sb1=[tuple(x) for x in subset]
    print(sb1)
    Shape_1 = calculatShapley(S1,sb1,[0,1,2])
    # print(getcoaltionlist())
    # item3='all'
    # if item3=='all':
    #     N = len(df)
    # else:
    #     N = int(item3)
    #
    # if sys.argv[3]=='all':
    #     N = len(df)
    # else:
    #     N = int(sys.argv[3])

    # df = df.sample(N)  # grab random subset (too slow otherwise)
    # print("!!!!!!!!!!!!!",df)
    # X, y = df.drop(targetcol, axis=1), df[targetcol]
    # COALITION=getcoaltionlist()
    # print("!!!!:",COALITION)

    #plot_anomalies(X, y, sample_size=sample_size, n_trees=n_trees, desired_TPR=desired_TPR, bins=15)
