import numpy as np
import random
def cost(pred,label):
    """The loss function of Factorization Machine
    input:pred(list)
          label(list)
    output:error(float)
    """
    m = len(pred)
    error = 0.0
    for i in range(m):
        error -= np.log(sig(pred[i]*label[i]))

    return error

def sig(x):
    return 1.0/(1+np.exp(-x))

# Initialize the weight of cross term
def initialize(n,k):
    """
    :param n:The num of features(int)
    :param k: (int)
    :return: v(mat) The weights of cross term
    """
    v = np.mat(np.zeros((n,k)))
    for i in range(n):
        for j in range(k):
            v[i,j] = random.normalvariate(0,0.2)
    return v

def stocGradAscent(feature,label,k,maxCycle,alpha):
    """
    :param feature:(mat)
    :param label: (mat)
    :param k: (int)
    :param maxCycle:(int)
    :param alpha: (float)
    :return: w0(float), w(mat),v(mat)
    """
    m,n = np.shape(feature)
    # Initialize parameters
    w = np.zeros((n,1))
    w0 = 0
    v = initialize(n,k)
    #Train Model
    for i in range(maxCycle):

        for x in range(m):
            inter1 = feature[x]*v # v*x
            inter2 = np.multiply(feature[x],feature[x]) * np.multiply(v,v) #dot product
            # The cross term
            interaction = np.sum(np.multiply(inter1,inter1)-inter2)*(1/2)
            p = w0 + feature[x]*w + interaction
            # update parameters
            w0 = w0 - alpha*(sig(p[0,0]*label[x])-1)*label[x]
            for y in range(n):
                if feature[x,y] !=0:
                    w[y,0] = w[y,0] - alpha*(sig(label[x]*p[0])-1)*label[x]*feature[x,y]
                    for j in range(k):
                        v[y,j] = v[y,j] - alpha*(sig(label[x]*p[0])-1)*label[x]*(feature[x,y]*inter1[0,j]-v[y,j]*feature[x,y]*feature[x,y])
        if i%100==0:
            print("\t-----------iter:",i,",cost:",cost(pred(feature,w0,w,v),label),"-------")

    return w0,w,v

def pred(feature,w0,w,v):
    m = np.shape(feature)[0]
    pred = []
    for i in range(m):
        inter1 = feature[i]*v
        inter2 = np.multiply(feature[i],feature[i])*np.multiply(v,v)
        interaction = np.sum(np.multiply(inter1,inter1)-inter2)/2
        p = w0 + feature[i]*w + interaction
        pre = sig(p[0,0])
        pred.append(pre)
    return pred

def Accuracy(pred,label):
    m = len(pred)
    error = 0
    for i in range(m):
        if float(pred[i])<0.5 and label[i] ==1.0:
            error+=1
        elif float(pred[i])>=0.5 and label[i] == -1.0:
            error+=1
        else:
            continue
    return 1-float(error)/m




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
        label_tmp.append(float(lines[-1])*2-1)

        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()
    return np.mat(feature_data),np.mat(label_data)



if __name__ == "__main__":
    print("-------1.load data--------")
    feature,label = load_data("D:/pyproj/ML-master/data/FMdata.txt")
    print("feature:",feature)
    print("label:",label)
    print("-------2.Train model-------")
    w0,w,v = stocGradAscent(feature,label,3,1000,0.01)
    print("w0:",w0)
    print("w:",w)
    print("v:",v)
    p = pred(feature,w0,w,v)
    print("------3.Accuracy----------")
    print(Accuracy(p,label))
