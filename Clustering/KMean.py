import numpy as np
import random

def load_data(path):
    f = open(path)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            data_tmp.append(float(lines[i]))
        data.append(data_tmp)
    return np.mat(data)


def Eu_dis(x, y):
    m, n = np.shape(x)
    dis = 0.0
    for i in range(n):
        dis += (x[0, i] - y[0, i]) ** 2
    return dis

# def IsNotEqual(mat1,mat2):
#     m,n = np.shape(mat1)
#     for it1 in range(m):
#         l = 0
#         for it2 in range(m):
#             # print(mat1[it1,:],"=========",mat2[it2,:])
#             if (mat1[it1,:] == mat2[it2,:]).all==True:
#                 l=1
#                 print("l##############:", l)
#                 break
#         if l==0:
#             # print("l##############:",l)
#             return 1
#     return 0





def KMean(data,k,mindis):
    m,n = np.shape(data)
    new_center = np.mat(np.zeros((k,n)))
    ran = random.sample(range(0,m-1),k)
    for it1 in range(k):
        new_center[it1,:]=data[ran[it1],:]
    label = np.mat(np.zeros((m,1)))
    center = np.mat(np.zeros((k, n)))
    while(True):
        center = new_center
        for it3 in range(m):
            M = mindis
            for it2 in range(k):
                dis = Eu_dis(data[it3,:],center[it2,:])
                # print("dis:",dis)
                if dis < M:
                    # print("it2:",it2)
                    M = dis
                    label[it3,0] = it2
        for it4 in range(k):
            count = 0
            c_mat = np.mat(np.zeros((m,n)))
            for it5 in range(m):
                if label[it5,0]==it4:
                    count += 1
                    c_mat[it5,:] = data[it5,:]
            new_center[it4,:] = np.sum(c_mat,axis=0)/count
        # print("newcenter:",new_center)
        # print("center:",center)
        if (new_center == center).all()==True:
            return (center,label)
        # print((new_center == center).all())




if __name__ == "__main__":
    print("\t----------1.load data---------------")
    data = load_data("D:/pyproj/ML-master/data/data_kmean.txt")
    print("data", data)
    # x = [1,2]
    # y = [1,0]
    # print("test_dis:",Eu_dis(np.mat(x),np.mat(y)))
    print("\t----------3.train model-------------")
    print("center:",KMean(data,4,100)[0])
    print("label:",KMean(data,4,100)[1])


