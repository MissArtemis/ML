import numpy as np

def load_data(file_name):
    """
    Load Training Set
    input: file_name(string)
    output: feature_data(mat)
            label_data(mat)
    """
    f = open(file_name)
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        label_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1) #constant term
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()
    return np.mat(feature_data),np.mat(label_data)

def Fderivative(feature,weight,label):
    F = -2*feature.T*(label-feature*weight)
    return F

def Sderivative(feature,weight,label):
    S = 2*feature.T*feature
    return S

def Mse(weight,feature,label):
    """
    :param weight:mat
    :param feature: mat
    :param label: mat
    :return: mse(float)
    """
    m = np.shape(feature)[0]
    mse = 0.0
    for i in range(m):
        mse += feature[i, :]*weight - label[i, 0]

    return mse

def global_newton(feature, label, maxCycle,sigma,delta):
    i = 0
    n = np.shape(feature)[1]
    weight = np.mat(np.ones((n,1)))
    while(i < maxCycle):
        g = Fderivative(feature,weight,label)
        G = Sderivative(feature,weight,label)
        d = -G.I*g
        m = Findmin(label,feature,g,d,weight,sigma,delta)
        weight = weight + (sigma**m)*d
        if i%10==0:
            mse = Mse(weight,feature,label)
            print("\t-----------i=",i," error:",mse[0,0])
        i = i+1
    return weight

def Findmin(label,feature,g,d,weight,sigma,delta):
    m = 0
    while True:
        f1 = (label - feature*(weight+(sigma**m)*d)).T*(label - feature*(weight+(sigma**m)*d))
        f2 = (label - feature*weight).T*(label - feature*weight)
        if f1[0,0] <= (f2 + delta*(sigma**m)*g.T*d)[0,0]:
            break
        else:
            m = m+1
    return m

if __name__ =="__main__":
    print("\t-----------------1.load data --------------")
    feature,label = load_data("D:/pyproj/ML-master/data/data_linear.txt")
    print("feature:",feature)
    print("label:",label)
    print("\t-----------------2.Training----------------")
    weight = global_newton(feature,label,500,0.1,0.1)
    print("weight:",weight)




