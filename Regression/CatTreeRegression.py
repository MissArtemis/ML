import numpy as np
import pickle
def load_data(inputpath):
    f = open(inputpath)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for i in lines:
            data_tmp.append(float(i))
        data.append(data_tmp)
    f.close()
    return data

class node:
    def __init__(self,left=None,right=None,col=-1,value=None,leaf=None):
        self.left = left
        self.right = right
        self.col = col
        self.value = value
        self.leaf = leaf

def leaf(data):
    data = np.mat(data)
    return np.mean(data[:, -1])

def var(dataset):
    data = np.mat(dataset)
    # print("data:",data[:,-1])
    var = np.var(data[:,-1]) * np.shape(data)[0]
    # print("var:",var)
    return var
def split_tree(data,col,value):
    t1 = []
    t2 = []
    for row in data:
        if row[col] >= value:
            t1.append(row)
        else:
            t2.append(row)
    return (t1,t2)

def build_tree(data,minSup,min_err):
    if len(data)<=minSup:
        return node(leaf=leaf(data))
    else:
        best_err = var(data)
        bestCut = None
        bestSet = None

        n = len(data[0])-1
        m = len(data)
        new_err = 0.0

        for col in range(n):
            value ={}
            for row in data:
                value[row[col]] = 1
            for key in value.keys():
                (t1,t2) = split_tree(data,col,key)
                # print("t1",t1)
                # print("t2",t2)
                if len(t1)>0 and len(t2)>0:
                    new_err = var(t1)+var(t2)

                if new_err<best_err and len(t1)>0 and len(t2)>0:
                    best_err = new_err
                    bestSet = (t1,t2)
                    bestCut = (col,key)
        if best_err>min_err:
            right = build_tree(bestSet[0],minSup,min_err)
            left = build_tree(bestSet[1],minSup,min_err)
            return node(left=left,right=right,col=bestCut[0],value=bestCut[1])
        else:
            return node(leaf=leaf(data))

def pred(sample,tree):
    if tree.leaf != None:
        return tree.leaf
    else:
        cut = sample[tree.col]
        branch =None
        if cut >tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return pred(sample,branch)

def cal_err(data,tree):
    m = len(data)
    n = len(data[0])-1
    err = 0.0
    for i in range(m):
        tmp=[]
        for j in range(n):
            tmp.append(data[i][j])
        pre = pred(tmp,tree)
        err+=(data[i][-1]-pre)**2
    return err/m

def save_model(tree,path):
    with open(path,'wb') as f:
        pickle.dump(tree,f)



if __name__ == "__main__":
    print("\t-------------1.load data--------------")
    data = load_data("D:/pyproj/ML-master/data/sine.txt")
    print("data:",data)
    print("\t-------------2.train model------------")
    tree = build_tree(data,minSup=10,min_err=0.1)
    print("\t-------------3.prediction--------------")
    err = cal_err(data,tree)
    print("Error:",err)
    print("\t------------4.save model----------------")
    save_model(tree,"D:/pyproj/ML-master/data/regression_tree.txt")