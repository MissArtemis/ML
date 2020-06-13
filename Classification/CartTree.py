import numpy as np


def load_data(input_path):
    f = open(input_path)
    data = []
    for line in f.readlines():
        data_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines)):
            data_tmp.append(float(lines[i]))
        data.append(data_tmp)
    f.close()
    return data


def Gini(data):
    label = []
    feature = []
    for row in data:
        feature_tmp = []
        label.append(row[-1])
        for it1 in range(len(row)-1):
            feature_tmp.append(row[it1])
        feature.append(feature_tmp)
    # return label,feature
    feature_mat = np.mat(feature)
    label_mat = np.mat(label)
    m,n = np.shape(feature_mat)
    label_set = set(label)
    # print(label_set)
    k = len(label_set)
    # return k
    cat = []
    for it2 in label_set:
        count = 0
        for it3 in range(m):
            if label[it3] == it2:
                count += 1
        cat.append(count)
    # print(cat)
    # return cat
    gini = 0.0
    for it4 in range(k):
        # print(cat[it4])
        gini += (cat[it4]/m)**2
    # print(gini)
    return 1-gini


class node:

    def __init__(self,left=None,right=None,leaf=None,col=-1,value=None):
        self.left = left
        self.right = right
        self.leaf = leaf
        self.col = col
        self.value = value

def split_tree(data,col,value):
    t1 = []
    t2 = []
    for row in data:
        if row[col] >= value:
            t1.append(row)
        else:
            t2.append(row)
    return (t1,t2)


def build_tree(data,minSample,min_gain):
    if len(data)<=minSample:
        return node(leaf = leaf(data))
    else:
        gini_old = Gini(data)
        bestGain = 0.0
        bestCol = None
        bestSets = None
        # label = []
        # feature = []
        # for row in data:
        #     feature_tmp = []
        #     label.append(row[-1])
        #     for it1 in range(len(row) - 1):
        #         feature_tmp.append(row[it1])
        #     feature.append(feature_tmp)
        # # return label,feature
        # feature_mat = np.mat(feature)
        # label_mat = np.mat(label)
        # m, n = np.shape(feature_mat)
        # label_set = set(label)
        # k = len(label_set)
        # cat =[]
        # for it2 in label_set:
        #     count = 0
        #     for it3 in range(m):
        #         if label[it3] == it2:
        #             count += 1
        #     cat.append(count)
        n = len(data[0])-1
        m = len(data)
        for col in range(n):
            col_value = {}
            for row in data:
                col_value[row[col]] = 1

            for key in col_value.keys():
                (t1,t2) = split_tree(data,col,key)
                gini_new = float(len(t1)*Gini(t1))/m + float(len(t2)*Gini(t2))/m
                Gain = gini_old -gini_new
                print(Gain)

                if Gain > bestGain and len(t1)>0 and len(t2)>0:
                    bestGain = Gain
                    bestCol = (col,key)
                    bestSets = (t1,t2)

        if bestGain>min_gain:
            print("Yes!")
            right = build_tree(bestSets[0],minSample,min_gain)
            left = build_tree(bestSets[1],minSample,min_gain)
            return node(col=bestCol[0],value=bestCol[1],right=right,left=left)
        else:
            return node(leaf=leaf(data))

def leaf(data):
    # print(data)
    data = np.mat(data)

    mean = np.mean(data[:,-1])
    if mean > 0.5:
        return 1
    else:
        return 0

def pred(sample,tree):
    if tree.leaf != None:
        return tree.leaf
    else:
        sample_col = sample[tree.col]
        branch = None

        if sample_col>=tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return pred(sample,branch)

def err(data,tree):
    m = len(data)
    n = len(data[0])-1
    err = 0
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(data[i][j])
        prediction = pred(tmp,tree)
        # print(prediction)
        if data[i][-1] != prediction:
            err += 1
    return err/m


if __name__ == "__main__":
    print("\t---------1.load data------------")
    data = load_data("D:/pyproj/ML-master/data/FMdata.txt")
    # data_test = load_data("D:/pyproj/ML-master/data/test_data_rf.txt")
    print("data:",data)
    # print("data_test",data_test)
    print("\t---------2.build tree-----------")
    tree = build_tree(data,1,0.003)
    print("\t---------3.Prediction-----------")
    err = err(data,tree)
    print("err:",err)
    # print("prediction:",pred)
    # print("Tree:",build_tree(data).leaf)
    # label,feature = Gini(data)
    # print("label",label)
    # print("feature",feature)
    # print("k:",Gini(data))
    # print("cat:",Gini(data))
    # print("gini:",Gini(data))



