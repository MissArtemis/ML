import numpy as np
def load_data(inputfile):
    """
    Load the data
    input:inputfile(string)
    output:feature(mat)
           label(mat)
           k(int):The num of label
    """
    f = open(inputfile)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        feature_tmp.append(1)
        lines = line.strip().split("\t")
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label.append(int(lines[-1]))
        feature.append(feature_tmp)
    f.close()
    return np.mat(feature), np.mat(label), len(set(label))
#Test
# f,l,k = load_data("D:/pyproj/ML-master/data/SoftInput.txt")
# print("f:",f,"l:",l,"k:",k)

def cost(weights,feature,label,k):
    """
    Calculate the loss function
    input:weight(mat)
          feature(mat)
          label(mat)
          k(int)
    output:cost(float)
    """
    cost = 0.0
    total = 0.0
    m,n = np.shape(feature)
    err = np.exp(feature*weights)
    total = err.sum(axis=1)
    err = np.log(err/total)
    for i in range(m):
        cost -= err[i,label[0,i]]

    return float(cost)/m


def gradientAscent(feature,label,k,maxCycle,alpha):
    m,n = np.shape(feature)
    weights = np.mat(np.ones((n,k)))
    i = 0
    while i <= maxCycle:
        total = 0.0
        i += 1
        if i%100==0 :
            print("\t---------iter:",i,"cost:",cost(weights,feature,label,k))
        error = np.exp(feature*weights)
        total= -error.sum(axis =1)
        #total = total.repeat(k,axis=1)
        error = error/total
        for j in range(m):
            error[j,label[0,j]] += 1
        weights = weights +(alpha/m)*feature.T*error
    return weights

def predict(feature,weights):
    h = feature*weights
    return h.argmax(axis=1) # return the label with the highest probability



if __name__=="__main__":
    print("------------1.Load data------------")
    feature,label,k = load_data("D:/pyproj/ML-master/data/SoftInput.txt")
    #weights = np.mat(np.ones((3, k)))
    #print(cost(weights,feature,label,k))
    print("------------2.Train model----------")
    weights = gradientAscent(feature,label,k,1000,0.1)
    print(weights)







