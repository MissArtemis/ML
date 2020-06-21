import numpy as np

def load_data(path):

    f = open(path)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            if lines[i] != "-":
                data_tmp.append(1)
            else:
                data_tmp.append(0)
        data.append(data_tmp)
    return np.mat(data)


def dt_dic(data):
    m,n = np.shape(data)
    data_dic = {}
    #For each user
    for i in range(m):
        data_tmp = {}
        for j in range(n):
            if data[i,j] !=0:
                data_tmp["D_"+str(j)] = data[i,j]
        data_dic["U_"+str(i)] = data_tmp

    #For each product
    for i in range(n):
        data_tmp = {}
        for j in range(m):
            if data[i,j] != 0:
                data_tmp["U_"+str(j)] = data[i,j]
        data_dic["D_"+str(i)] = data_tmp
    return data_dic

def PageRank(data_dic,alpha,maxCycle,user,lam):
    #Initial Rank
    rank = {}
    #Start from User:user
    for x in data_dic.keys():
        rank[x] = 0
    rank[user] = 1

    iter = 0

    while iter <= maxCycle:
        tmp = {}
        for x in data_dic.keys():
            tmp[x] = 0
        for i, ri in data_dic.items():
            # print("i:",i)
            # print("ri:",ri)
            for j in ri.keys():
                if j not in tmp:
                    tmp[j] = 0
                tmp[j] += alpha*rank[i]/(float(len(ri)))
                if j == user:
                    tmp[j] += (1-alpha)
        check = []
        for k in tmp.keys():
            check.append(tmp[k]-rank[k])
        if sum(check) <= lam:
            break
        rank = tmp
        iter +=1
    return rank

def Rec(data_dic,rank,user):
    items_dic = {}
    #Find already ranked item for user:user
    items = []
    for k in data_dic[user].keys():
        items.append(k)

    #rating the items from rank

    for k in rank.keys():
        if k.startswith("D_"):
            if k not in items:
                items_dic[k] = rank[k]

    result = sorted(items_dic.items(),key=lambda d:d[1],reverse=True)
    return result




if __name__ == "__main__":
    print("\t-------------1.load data-----------")
    data = load_data("D:/pyproj/ML-master/data/data_pr.txt")
    print("data:",data)
    print("\t-------------2.Create Data_dic-------")
    data_dic = dt_dic(data)
    print("data_dic:",data_dic)
    print("\t------------3.Page Rank -----------")
    rank = PageRank(data_dic,0.85,100,"U_0",0.0001)
    print("Rank:",rank)
    print("\t------------4.Recommend------------")
    rec = Rec(data_dic,rank,"U_0")
    print("result:",rec)