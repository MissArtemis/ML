#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:50:18 2020

@author: zhangjinghang
"""
import itertools

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import DataFrame
import copy



class AprioriSeq():
    '''
    Self Defined Apriori based Sequence mining
    '''
    
    def __init__(self,data:DataFrame,col,minSupport = 0.01,maxLength = 5,minLength = 1):
        self.data = data 
        self.col = col
        self.minSupport = minSupport
        self.maxLength = maxLength
        self.minLength = minLength

    def getTotalNum(self):
        col = self.col
        data = self.data
        data=data.withColumn(col,data[col].cast(ArrayType(StringType())))
        data = data.select(col)
        num = data.count()
        self.threshold = num*self.minSupport
        print(self.threshold)
        return num
    
    def getItemsSet(self):
        data = self.data
        col = self.col
        num = self.getTotalNum()
        data = data.withColumn("items",explode(col))
        countBase = data.groupBy("items").count()
        countBase = countBase.withColumn("items",split(countBase["items"]," "))
        filterBase = countBase.where(countBase['count']>=int(self.threshold)).orderBy("count",ascending=False)
        return filterBase

    def isSubsequence(self,sequence1:list, sequence2:list):
        if len(sequence1) > len(sequence2):
            return "False"
        else:
            if sequence1!=[]:
                if sequence1[0] in sequence2:
                    ind = sequence2.index(sequence1[0])
                    sequence2 = sequence2[ind:]
                    sequence1 = sequence1[1:]
                    return self.isSubsequence(sequence1,sequence2)
                else:
                    return "False"
            else:
                return "True"
    
    def generateCand(self,seq11:list,seq22:list):
        seq1 = copy.deepcopy(seq11)
        seq2 = copy.deepcopy(seq22)
        if len(seq1)!=len(seq2):
            # print("The candidate is not from the same dim")
            return []
        else:
            body1 = seq1[1:]
            body2 = seq2[:-1]
            if body1 == body2:
                # print("Yes")
                seq1.append(seq2[-1])

                return seq1
            else:
                print("Not from same body")
                return []

    def combo(self,list):
        cob = []
        for i in range(len(list)):
            for j in range(len(list)):
                cob.append([list[i],list[j]])
        return cob



    def fit(self):
        k=1
        countdtbase = self.getItemsSet()
        self.freqSeq = countdtbase
        data = self.data.select(self.col)
        ItemSet = set()
        # return ItemSet
        base = countdtbase.select("items").rdd.map(lambda x: x[0]).collect()
        dataPanda = data.toPandas()
        while k<self.maxLength:
            cand_all = self.combo(base)
            print(cand_all)

            cand_select = []
            cand = []
            for cd in cand_all:
                print(cd[0],cd[1])
                newCand = self.generateCand(cd[0],cd[1])
                # print(newCand)
                if newCand == []:
                    continue
                # print(newCand)
                dataPanda['result'] = dataPanda[self.col].apply(lambda x:self.isSubsequence(newCand,x))
                count = (dataPanda['result']=='True').sum()
                print(newCand,":",count)
                if count >= self.threshold:
                    cand_select.append(str(newCand))
                    if newCand not in cand:
                        cand.append(newCand)
                    # print(cand_select)

            if k >= self.minLength:
                base = set(cand_select)
                ItemSet.update(base)
                # print(ItemSet)
                base = cand
                # print(base)
            k += 1

        self.result = ItemSet

        return ItemSet



    def checkNoise(self):
        countdtbase = self.getItemsSet()
        data = self.data.select(self.col)
        base = countdtbase.select("items").rdd.map(lambda x: x[0]).collect()
        # print(base)
        one_all = self.combo(base)
        print(one_all)
        dataPanda = data.toPandas()
        noise = []
        selected = []
        for one in one_all:
            two_all = self.generateCand(one[0],one[1])
            print(one)
            dataPanda['result'] = dataPanda[self.col].apply(lambda x: self.isSubsequence(two_all, x))
            count = (dataPanda['result'] == 'True').sum()
            if count >= self.threshold:
                if one[0]==one[1]:
                    noise.append(two_all)
                elif [one[1],one[0]] in selected:
                    noise.append(two_all)
                else:
                    selected.append(two_all)
        self.noise = noise
        return noise


if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("SeqMining").getOrCreate()
    data = spark.read.csv("D:/pyproj/ML-master/SequenceMining/SeqMining/path.csv",sep="\t")
    # data.show()
    columns=["hashedIpAddress","timestamp","durationInSec","path","rating"]
    data=data.toDF(*columns)
    # data.show()
    data = data.withColumn("path",split(data['path'],";"))
    # data.show()
    data = data.withColumn("path",explode(data['path']))
    # data.show()
    data = data.where(data['path']!="<")
    data = data.groupBy(data['hashedIpAddress']).agg(collect_list(data['path']))
    data = data.withColumnRenamed("collect_list(path)","path")
    data.show()
    model = AprioriSeq(data,"path",minSupport=0.01,maxLength=3)
    # noise = model.checkNoise()
    # print(noise)
    pattern = model.fit()
    # pattern.show(20,False)
    print(pattern)
    # count = model.getItemsSet()
    # count.show()
    # ItemSet=set(count.select("items").rdd.map(lambda x:str(x[0])).collect())
    # print(list(itertools.combinations(ItemSet,2)))
    # tuple = (["a"],["b"])
    # a = tuple[0]
    # b = tuple[1]
    # print("a:",a," b:",b)



    # seq1 = ["c","b"]
    # seq2 = ["a","b","c"]
    # seq3 = ["b","c","d"]
    # seq4 = ["b","c"]
    # seq5 = ["b","d"]
    # print("seq2 body: ", seq2[1:])
    # print("seq3 body:", seq3[:-1])
    # print(model.isSubsequence(seq1,seq2))
    # print(model.isSubsequence(seq4,seq2))
    # print(model.isSubsequence(seq5,seq3))
    # print(model.generateCand(seq2,seq3))

    # num = model.getTotalNum()
    # print(num)
    # itemset = model.getItemsSet()
    # itemset.show()
    
    
    


    
    