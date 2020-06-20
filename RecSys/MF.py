import numpy as np

def load_data(path):
    f = open(path)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            if lines[i] != "-":
                data_tmp.append(float(lines[i]))
            else:
                data_tmp.append(0.0)
        data.append(data_tmp)
    return np.mat(data)



def MF(data,k,maxIter,alpha,beta):
    m,n  = np.shape(data)
    p = np.mat(np.random.random((m,k)))
    q = np.mat(np.random.random((k,n)))
    iter = 0
    err = 0.0
    while(iter<maxIter):
        for i in range(m):
            for j in range(n):
                for x in range(k):
                    p[i,x] = p[i,x] + alpha*(2*(data[i,j]-p[i,:]*q[:,j])*q[x,j]-beta*p[i,x])
                    q[x,j] = q[x,j] + alpha*(2*(data[i,j]-p[i,:]*q[:,j])*p[i,x]-beta*q[x,j])
        iter += 1
        if iter%100 == 0:
            err = 0.0
            for i in range(m):
                for j in range(n):
                    err += (data[i,j]-p[i,:]*q[:,j])**2
                    for x in range(k):
                        err += 1/2*beta*(p[i,x]**2+q[x,j]**2)
            print("\t---------iter=:",iter)
            print("err:",err)
    return p,q

def pred(data,p,q):
    m,n = np.shape(data)
    for i in range(m):
        for j in range(n):
            if data[i,j]==0:
                data[i,j] = p[i,:]*q[:,j]
                if data[i,j]<0:
                    data[i,j]=0
    return data



if __name__ == "__main__":
    print("\t--------------1.load data------------------")
    data = load_data("D:/pyproj/ML-master/data/data_MF.txt")
    print("data:", data)
    print("\t-------------2.training--------------------")
    p,q = MF(data,3,5000,0.1,0.02)
    print("p:",p)
    print("q:",q)
    print("\t------------3.prediction-------------------")
    pred = pred(data,p,q)
    print("prediction:",pred)


