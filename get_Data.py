import scipy.io
import numpy as np


def getData(s,ss):
    mat = []
    for i in range(1,21):  
        m = scipy.io.loadmat(s +ss + str(i) +  ".mat")
        mat.append(m)
    X = []
    Y = []
    I = []
    for i in range(0,20):
        M = mat[i]
        dd = ss + str(i+1)
        M = M [dd]
        X.append(M["X"][0][0]["Data"][0])
        Y.append(M["Y"][0][0]["Data"][0])
        

    XData = np.arange(0)
    YData = np.arange(0)
    IData = np.arange(0)

    for i in range(0,20):
        XData = np.append(XData,X[i][1][0])
        YData = np.append(YData, Y[i][6][0])
    return [XData,YData]

def getDataI(s,ss):
    mat = []
    for i in range(1,21):  
        m = scipy.io.loadmat(s +ss + str(i) +  ".mat")
        mat.append(m)
    X = []
    Y = []
    #I = []
    for i in range(0,20):
        M = mat[i]
        dd = ss + str(i+1)
        M = M [dd]
        X.append(M["X"][0][0]["Data"][0])
        Y.append(M["Y"][0][0]["Data"][0])
       # I.append(I,M[0][0][2][0][1][2][0])

    XData = np.arange(0)
    YData = np.arange(0)
    IData = np.arange(0)

    for i in range(0,20):
        XData = np.append(XData,X[i][1][0])
        YData = np.append(YData, Y[i][6][0])
        IData = np.append(IData,M[0][0][2][0][1][2][0])
    return [XData,YData,IData]

def getSample(s,ss):
    m = scipy.io.loadmat(s+ss+".mat")
    M = m[ss]
    X = np.arange(0)
    Y = np.arange(0)
    X = np.append(X,M["X"][0][0]["Data"][0][1][0])
    Y = np.append(Y,M["Y"][0][0]["Data"][0][6][0])
    return [X,Y]

def getSampleI(s,ss):
    m = scipy.io.loadmat(s+ss+".mat")
    M = m[ss]
    X = np.arange(0)
    Y = np.arange(0)
    I = np.arange(0)
    X = np.append(X,M["X"][0][0]["Data"][0][1][0])
    Y = np.append(Y,M["Y"][0][0]["Data"][0][6][0])
    I = np.append(I,M[0][0][2][0][1][2][0])
    return [X,Y,I]
    
#YData4 = fft(YData4)
N = 1000
T = 1.0 / 400.0
#lx = fftfreq(N, T)[:N//2]
X1,Y1 = getSample("bearing/K005/","N15_M07_F10_K005_1")
X2,Y2 = getSample("bearing/KA05/","N15_M07_F10_KA05_1")
X3,Y3 = getSample("bearing/KI05/","N15_M07_F10_KI05_1")
X4,Y4 = getSample("bearing/KB27/","N15_M07_F10_KB27_1")

X1 = X1[0:256000]
X1 = np.linspace(0,4,256000)
Y1 = Y1[0:256000]
Y2 = Y2[0:256000]
Y3 = Y3[0:256000]
Y4 = Y4[0:256000]

XData1, YData1 = getData("bearing/K005/","N15_M07_F10_K005_")
XData2, YData2 = getData("bearing/KA05/","N15_M07_F10_KA05_")
XData3, YData3 = getData("bearing/KI05/","N15_M07_F10_KI05_")    
XData4, YData4 = getData("bearing/KB27/","N15_M07_F10_KB27_") 
lx = np.linspace(0,80, 5120000)
XData1 = XData1[0:len(lx)]
YData1 = YData1[0:len(lx)]
YData2 = YData2[0:len(lx)]
YData3 = YData3[0:len(lx)]
YData4 = YData4[0:len(lx)]
