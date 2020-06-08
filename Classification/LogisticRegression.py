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

def sigmoid(x):
    """
    Sigmoid Function
    input:x(mat):feature*w
    output: sigmoid(x)(mat)
    """
    return 1/(1+np.exp(-x))

def graddcent(feature,label,maxCycle,alpha):
    """
    Using Gradient Descent to estimate parameters of LR Model
    :param feature: mat
    :param label: mat
    :param maxCycle: int
    :param alpha: float
    :return: weight(mat)
    """
    n = np.shape(feature)[1] # The num of features
    weight = np.mat(np.ones((n,1))) # Initialize weight
    i = 0
    while i <= maxCycle:
        i = i + 1
        h = sigmoid(feature*weight)
        err = label - h
        if i % 100 == 0:
            print("---------iter = "+str(i)+", train error rate = " + str(err_rate(label,h)))
        weight = weight - alpha*(-feature.T)*err
    return weight

def err_rate(label,h):
    """

    :param label:(mat)
    :param h: (mat)
    :return: err/m(float)
    """
    m = np.shape(h)[0]
    sum = 0
    for i in range(m):
        if h[i,0]>0 and (1-h[i,0])>0: #In case h[i,0] = 0 or h[i,0] = 1
            sum -= (label[i,0]*np.log(h[i,0])+(1-label[i,0])*np.log(1-h[i,0]))
        else:
            sum = sum+0
    return sum/m

def save_model(file_name,weight):
    """

    :param file_name:string
    :param weight:mat
    :return:
    """
    m = np.shape(weight)[0]
    f_write = open(file_name,"w")
    w_array = []
    for i in range(m):
        w_array.append(str(weight[i,0]))
    f_write.write("\t".join(w_array))
    f_write.close()

def predict(feature_dt,weight):
    """

    :param feature_dt:mat
    :param weight: mat
    :return: h(mat):y_hat
    """
    h = sigmoid(feature_dt*weight)
    m = np.shape(h)[0]
    for i in range(m):
        if h[i,0]<0.5:
            h[i,0] = 0
        else:
            h[i,0] = 1
    return h


if __name__ == "__main__":

# Test load_data
#     f,l = load_data("D:/pyproj/ML-master/data/LR/data.txt")
#     print("feature: ",f," label:",l)
#     weight = np.mat(np.ones((10,1)))
#     print(weight)
    print("--------1.Load Data---------")
    feature,label = load_data("D:/pyproj/ML-master/data/LR/data.txt")
    print("--------2.Training----------")
    weight = graddcent(feature,label,1000,0.01)
    print("--------3.Save Model--------")
    save_model("LR_model.txt",weight)