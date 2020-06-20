import numpy as np
def least_square(feature,label):
    """
    :param feature:mat
    :param label: mat
    :return: weight(mat)
    """
    weight = (feature.T*feature).I *feature.T*label
    return weight

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

if __name__ =="__main__":
    print("\t-----------------1.load data --------------")
    feature,label = load_data("D:/pyproj/ML-master/data/SoftInput.txt")
    print("feature:",feature)
    print("label:",label)
    print("\t-----------------2.Training----------------")
    weight = least_square(feature,label)
    print("weight:",weight)
    print("\t-----------------3.Error-------------------")
    mse = Mse(weight,feature,label)
    print("MSE:",mse)