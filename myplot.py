from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,recall_score,cohen_kappa_score,roc_curve,precision_score
from adtk import detector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

my_dec = detector
array_true = []
s = []

def calcRocC(my_dec,c=20):
    
        
    curve_tpr = np.zeros(0)
    curve_fpr = np.zeros(0)
    curve_tresh = np.zeros(0)
    anomalies = np.zeros(0)
    c_array = np.linspace(0,3,c)
    for i in range(0,c):
        
        ad = my_dec(c=c_array[i])
        anomalies = ad.fit_detect(s)
        fpr,tpr,tresh= roc_curve(array_true,calcMetric(array_true,anomalies.iloc[0:,1],False))
        curve_tpr = np.append(curve_tpr,tpr)
        curve_fpr= np.append(curve_fpr,fpr)
        curve_tresh = np.append(curve_tresh,tresh)

    zz = list(zip(curve_tpr,curve_fpr))
    s_zz = sorted(zz,key = lambda x:x[1])
    curve_tpr,curve_fpr = zip(*s_zz)


    return curve_fpr,curve_tpr,curve_tresh

def calcRocW(my_dec,c=20):
    
        
    curve_tpr = np.zeros(0)
    curve_fpr = np.zeros(0)
    curve_tresh = np.zeros(0)
    anomalies = np.zeros(0)
    c_array = np.linspace(0,3,c)
    for i in range(0,c):
        
        ad = my_dec(c=c_array[i],window=50)
        anomalies = ad.fit_detect(s)
        fpr,tpr,tresh= roc_curve(array_true,calcMetric(array_true,anomalies.iloc[0:,1],False))
        curve_tpr = np.append(curve_tpr,tpr)
        curve_fpr= np.append(curve_fpr,fpr)
        curve_tresh = np.append(curve_tresh,tresh)

    zz = list(zip(curve_tpr,curve_fpr))
    s_zz = sorted(zz,key = lambda x:x[1])
    curve_tpr,curve_fpr = zip(*s_zz)


    return curve_fpr,curve_tpr,curve_tresh

def calcRocTresh(my_dec,c=20):
    
        
    curve_tpr = np.zeros(0)
    curve_fpr = np.zeros(0)
    curve_tresh = np.zeros(0)
    anomalies = np.zeros(0)
    c_array = np.linspace(0,3,c)
    for i in range(0,c):
        
        ad = my_dec(high=c_array[i],low=-c_array[i])
        anomalies = ad.detect(s)
        
        fpr,tpr,tresh= roc_curve(array_true,calcMetric(array_true,anomalies.iloc[0:,1],False))
        curve_tpr = np.append(curve_tpr,tpr)
        curve_fpr= np.append(curve_fpr,fpr)
        curve_tresh = np.append(curve_tresh,tresh)

    zz = list(zip(curve_tpr,curve_fpr))
    s_zz = sorted(zz,key = lambda x:x[1])
    curve_tpr,curve_fpr = zip(*s_zz)


    return curve_fpr,curve_tpr,curve_tresh

def calcRocQuantile(my_dec,c=20):
    
        
    curve_tpr = np.zeros(0)
    curve_fpr = np.zeros(0)
    curve_tresh = np.zeros(0)
    anomalies = np.zeros(0)
    c_array = np.linspace(0,1,c)
    for i in range(0,c):
        
        ad = my_dec(high=c_array[i],low=0.01)
        anomalies = ad.fit_detect(s)
        
        fpr,tpr,tresh= roc_curve(array_true,calcMetric(array_true,anomalies.iloc[0:,1],False))
        curve_tpr = np.append(curve_tpr,tpr)
        curve_fpr= np.append(curve_fpr,fpr)
        curve_tresh = np.append(curve_tresh,tresh)

    zz = list(zip(curve_tpr,curve_fpr))
    s_zz = sorted(zz,key = lambda x:x[1])
    curve_tpr,curve_fpr = zip(*s_zz)


    return curve_fpr,curve_tpr,curve_tresh

def calcRocGen(my_dec,c=20):
    
        
    curve_tpr = np.zeros(0)
    curve_fpr = np.zeros(0)
    curve_tresh = np.zeros(0)
    anomalies = np.zeros(0)
    c_array = np.linspace(0,6,c)
    for i in range(0,c):
        
        ad = my_dec(alpha = c_array[i])
        anomalies = ad.fit_detect(s)
        
        fpr,tpr,tresh= roc_curve(array_true,calcMetric(array_true,anomalies.iloc[0:,1],False))
        curve_tpr = np.append(curve_tpr,tpr)
        curve_fpr= np.append(curve_fpr,fpr)
        curve_tresh = np.append(curve_tresh,tresh)

    zz = list(zip(curve_tpr,curve_fpr))
    s_zz = sorted(zz,key = lambda x:x[1])
    curve_tpr,curve_fpr = zip(*s_zz)


    return curve_fpr,curve_tpr,curve_tresh

data = pd.DataFrame
def myplot(tt,data, anomalies):
    #print(len(tt),len(data))
    ano1 = anomalies.iloc[0:,0]
    ano2 = anomalies.iloc[0:,1]
    dataA = data.iloc[0:,0]
    dataB = data.iloc[0:,1]
    aa = []
    at = []
    bb = []
    bt = []
    
    for i in range(0,len(tt)):
        if (ano1[i]==True):
            aa = aa.append(aa,dataA[i])
            at = at.append(at,tt[i])
        if (ano2[i]==True):
            bb = bb.append(bb,dataB[i])
            bt = bt.append(bt,tt[i])
    
    fig, (ax1,ax2) = plt.subplots(nrows=2,ncols=1,figure= (10,5))
    ax1.plot(tt,data["healthy"],marker=".",color = "blue",label="Gesund",zorder=1)
    ax2.plot(tt,data["broken"],marker=".",color= "orange",label="Beschädigt",zorder=1)
    ax1.scatter(at,aa,marker = "o",color="red",label="Ausreißer",zorder=2)
    ax2.scatter(bt,bb,marker="o",color="red",label ="Ausreißer",zorder=2)
    ax1.set_ylabel("Beschleunigung in [m/$s^2$]")
    ax1.set_xlabel("Zeit in [s]")
    ax2.set_ylabel("Beschleunigung in [m/$s^2$]")
    ax2.set_xlabel("Zeit in [s]")
    ax1.legend()
    ax2.legend()
    plt.show()

def myplot1(tt,data, anomalies):
    ano1 = anomalies.iloc[0:]  
    dataA = data.iloc[0:]
    aa = []
    at = []
    anom = 0  
    for i in range(0,len(tt)):
        if (ano1[i]==True):
            aa = aa.append(aa,dataA[i])
            at = at.append(at,tt[i])
            anom = anom +1
    
    fig, (ax1) = plt.subplots(nrows=2,ncols=1,figure= (10,5))
    ax1.plot(tt,data["healthy"],marker=".",color = "blue",label="gesund",zorder=1)
    ax1.scatter(at,aa,marker = "o",color="red",label="Ausreißer",zorder=2)
    ax1.set_ylabel("Beschleunigung in [m/s^2]")
    ax1.set_xlabel("Zeit in [s]")
    ax1.legend()
    plt.show()
    return anom
    

def myplotmulti(tt,data, anomalies):
    print(len(tt),len(data))
    ano1 = anomalies.iloc[0:]
    #ano2 = anomalies.iloc[0:,1]
    dataA = data.iloc[0:,0]
    #dataB = data.iloc[0:,1]
    aa = np.empty(0)
    at = np.empty(0)
    #bb = np.empty(0)
    #bt = np.empty(0)
    
    for i in range(0,len(tt)):
        if (ano1[i]==True):
            aa = np.append(aa,dataA[i])
            at = np.append(at,tt[i])
        #if (ano2[i]==True):
         #   bb =np.append(bb,dataB[i])
          #  bt = np.append(bt,tt[i])
    
    fig, (ax1) = plt.subplots(nrows=1,ncols=1,figure= (10,5))
    ax1.plot(tt,data["healthy"],marker=".",color = "blue",label="gesund",zorder=1)
    ax1.plot(tt,data["broken"],marker=".",color= "orange",label="beschädigt",zorder=1)
    ax1.scatter(at,aa,marker = "o",color="red",label="Ausreißer",zorder=2)
    #ax2.scatter(bt,bb,marker="o",color="red",label ="Ausreißer",zorder=2)
    ax1.set_ylabel("Beschleunigung in [m/s^2]")
    ax1.set_xlabel("Zeit in [s]")
    #ax2.set_ylabel("Beschleunigung in [m/s^2]")
    #ax2.set_xlabel("Zeit in [s]")
    ax1.legend()
    #ax2.legend()
    plt.show()


def calcMetric(array_true, array_pred,ff=False):
    tt = np.zeros(len(array_pred))
    for i in range(0,len(array_pred)):
        if array_pred[i]== True:
            tt[i] = True
        else:
            tt[i]= False
    array_pred = tt
    if ff == True:
        print("Accuracy score   = "  ,accuracy_score(array_true, array_pred))
        print("Precision score  = "  ,precision_score(array_true, array_pred))
        print("Recall score     = "  ,recall_score(array_true, array_pred))
        print("F1 score         = "  ,f1_score(array_true, array_pred))
        print("Cohens score     = "  ,cohen_kappa_score(array_true, array_pred))
    return array_pred





   