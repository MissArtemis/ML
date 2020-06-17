import numpy as np

def load_data(path):
    f = open(path)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            if lines[i]=="-":
                lines[i] = 0
            data_tmp.append(float(lines[i]))
        data.append(data_tmp)
    f.close()
    return np.mat(data)

def cossim(x,y):
    return (x*y.T)/(np.sqrt(x*x.T)*np.sqrt(y*y.T))

def cov_mat(data):
    m,n = np.shape(data)
    cov = np.mat(np.zeros((m,m)))
    for i in range(m):
        for j in range(m):
            if i!=j:
                cov[i,j] = cossim(data[i,:],data[j,:])
                cov[j,i] = cov[i,j]
            else:
                cov[i,j] = 0
    return cov

def usr_rec(data,cov):
    m,n = np.shape(data)
    for i in range(m):
        for j in range(n):
            if data[i,j] == 0:
                for k in range(m):
                    data[i,j] += data[k,j]*cov[i,k]
    return data


if __name__ == "__main__":
    print("\t-----------1.load data-------------")
    data = load_data("D:/pyproj/ML-master/data/data_CF.txt")
    print("data:", data)
    print("\t-----------2.train model")
    cov = cov_mat(data)
    print("covariance",cov)
    print("\t-----------3.prediction display------")
    print("pred",usr_rec(data,cov))
