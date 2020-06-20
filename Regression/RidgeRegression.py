import numpy as np


def RidgeRegression(feature, label, lam):
    n = np.shape(feature)[1]
    weight = (feature.T * feature + lam * np.mat(np.eye(n))).I * feature.T * label
    return weight


def Mse(label, feature, weight):
    m = np.shape(feature)[0]
    err = 0.0
    for i in range(m):
        err += label[i, 0] - feature[i, :] * weight
    return err

def load_data(input):
    f = open(input)
    label = []
    feature = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    return np.mat(feature),np.mat(label).T

if __name__ == "__main__":
    print("\t----------------1.load data---------------")
    feature,label = load_data("D:/pyproj/ML-master/data/meterial.txt")
    print("feature:",feature)
    print("label:",label)
    print("\t---------------2.training-----------------")
    weight = RidgeRegression(feature,label,0.01)
    print("weight:",weight)
    print("\t---------------3.MSE----------------------")
    err = Mse(label,feature,weight)
    print("err:",err)