import numpy as np
import math

MinPts = 7


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


def Dist(data):
    m, n = np.shape(data)
    dist = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(m):
            dist[i, j] = np.sqrt((data[i, :] - data[j, :]) * (data[i, :] - data[j, :]).T)

    return np.mat(dist)


def find_points(dist_D, eps):
    pts_index = []
    n = np.shape(dist_D)[1]
    for i in range(n):
        if dist_D[0, i] < eps:
            pts_index.append(i)
    return pts_index


def epsilon(data, MinPts):
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps


def DBSCAN(data, eps, MinPts):
    m, n = np.shape(data)
    types = np.mat(np.zeros((1, m)))
    sub_class = np.mat(np.zeros((1, m)))
    # Core pts labels 1, border points labels 0, and noise points labels -1
    deal = np.mat(np.zeros((1, m)))
    # 1 means done and 0 means undone
    dist = Dist(data)
    label = 1

    for i in range(m):
        if deal[0, i] == 0:
            D = dist[i,]
            pt_index = find_points(D, eps)

            # The border points
            if len(pt_index) > 1 and len(pt_index) < MinPts + 1:
                types[0, i] = 0
                sub_class[0, i] = 0

            # The noise points
            if len(pt_index) == 1:
                types[0, i] == -1
                sub_class[0, i] == -1
                deal[0, i] == 1

            # The core points:
            if len(pt_index) > MinPts:
                types[0, i] = 1
                for x in pt_index:
                    sub_class[0, x] = label
                # Judge whether the points are density reachable
                while len(pt_index)>0:
                    deal[0,pt_index[0]]=1
                    D = dist[pt_index[0],]
                    tmp = pt_index.pop(0)
                    pt_index1 = find_points(D,eps)

                    if len(pt_index1)>1:
                        for i in pt_index1:
                            sub_class[0,i] = label
                        if len(pt_index1)>=MinPts+1:
                            types[0,tmp] = 1
                        else:
                            types[0,tmp] = 0
                        for j in range(len(pt_index1)):
                            if deal[0,pt_index1[j]] ==0:
                                deal[0, pt_index1[j]]=1
                                pt_index.append(pt_index1[j])
                                sub_class[0,pt_index1[j]] = label
                label += 1
    pt_index2 = ((sub_class == 0).nonzero())[1]
    for x in pt_index2:
        sub_class[0, x] = -1
        types[0, x] = -1

    return types, sub_class


if __name__ == "__main__":
    print("\t-----------1.load data-----------")
    data = load_data("D:/pyproj/ML-master/data/data_dbscan.txt")
    print("data:", data)
    print("\t-----------2.cal dist------------")
    dist = Dist(data)
    print("dist:", dist)
    print("\t-----------3.DBSCAN--------------")
    ep = epsilon(data, MinPts)
    typ, lb = DBSCAN(data, ep, MinPts)
    print("typ:", typ)
    print("label",lb)
