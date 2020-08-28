# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 10:07:47 2020

@author: z677159
"""
from pyspark.sql import SparkSession
from InfoValue.pysparkIV import pysparkIV


if __name__ =="__main__":
    spark = SparkSession.builder.appName("DataProcessing").getOrCreate()
    data = spark.read.csv("/Users/zhangjinghang/Desktop/Lab/data-final.csv",inferSchema=True,header=True,sep="\t")
    #feature = data.drop("           dateload").columns
    #print(data.columns)
    # feature = ['ratings','family']
    feature = data.select("EXT1","EXT2","EXT3","EXT4","EXT5","EXT6","EXT7","EXT8","EXT9","EXT10","EST1","EST2","EST3","EST4","EST5","EST6","EST7","EST8","EST9","EST10","AGR1","AGR2","AGR3","AGR4","AGR5","AGR6","AGR7","AGR8","AGR9","AGR10","CSN1","CSN2","CSN3","CSN4","CSN5","CSN6","CSN7","CSN8","CSN9","CSN10","OPN6","OPN7","OPN8","OPN9","OPN10","EXT1_E","EXT2_E","EXT3_E","EXT4_E","EXT5_E","EXT6_E","EXT7_E","EXT8_E","EXT9_E","EXT10_E","EST1_E","EST2_E","EST3_E","EST4_E","EST5_E").columns
    pyIV = pysparkIV(data,'country',feature,'HK',0.1)
    pyIV.fit()
    IV_rs=pyIV.agg_iv()
    print(IV_rs)
    
    
    
    