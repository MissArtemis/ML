import numpy as np
def Newton(feature,label,maxCycle):
    """
    :param feature:mat
    :param label: mat
    :param maxCycle: int
    :param e: int
    :return: weight(mat)
    """
    i = 0
    d = 0
    m,n = np.shape(feature)
    weight = np.mat(np.ones((n,1)))
    while(i < maxCycle):
        g = Fderivative(feature,weight,label)
        G = Sderivative(feature,weight,label)
        d =-G.I*g
        weight = weight + d
        i+=1
        if i%10==0:
            mse = Mse(weight,feature,label)
            print("\t-----------i=",i," error:",mse[0,0])
    return weight

def Fderivative(feature,weight,label):
    F = -2*feature.T*(label-feature*weight)
    return F

def Sderivative(feature,weight,label):
    S=2*feature.T*feature
    return S

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


if __name__ =="__main__":
    print("\t-----------------1.load data --------------")
    feature,label = load_data("D:/pyproj/ML-master/data/data_linear.txt")
    print("feature:",feature)
    print("label:",label)
    print("\t-----------------2.Training----------------")
    weight = Newton(feature,label,100)
    print("weight:",weight)
    # print("\t-----------------3.Error-------------------")
    # mse = Mse(weight,feature,label)
    # print("MSE:",mse)