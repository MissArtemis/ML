# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:12:34 2020

@author: z677159
"""

import math
from pyspark.sql import DataFrame
from pyspark.sql.functions import *

class pysparkIV(object):
    def __init__(self,df:DataFrame,label:str,feature:[str],target:str,theta:float):
        self.df = df
        self.label = label
        self.feature = feature
        self.target = target
        self.result = {}
        self.theta = theta
    
    def fit(self):
        flg1 = self.total_flg1()
        flg0 = self.total_flg0()
        for col in self.feature:
            data = self.df.select(col)
            cats = data.distinct().collect()
            
            for row in cats:
                cat = row[col]
                flg1_count = self.count_flg1(col,cat)
                flg0_count = self.count_flg0(col,cat)
                
                if flg1_count==0:
                    flg1_count = self.theta
                
                if flg0_count==0:
                    flg0_count = self.theta
                
                flg1_dist = float(flg1_count)/flg1
                flg0_dist = float(flg0_count)/flg0
                
                self.build_data(col,cat,flg1_dist,flg0_dist)
    
    def total_flg1(self):
        return self.df.select(self.label).where(self.df[self.label]==self.target).count()
    
    def total_flg0(self):
        return self.df.select(self.label).where(self.df[self.label]!=self.target).count()
    
    def count_flg1(self,col:str,cat:str):
        return self.df.select(col,self.label).where((self.df[col]==cat)&(self.df[self.label]==self.target)).count()
    
    def count_flg0(self,col:str,cat:str):
        return self.df.select(col,self.label).where((self.df[col]==cat)&(self.df[self.label]!=self.target)).count()
    
    def build_data(self,col,cat,flg1_dist,flg0_dist):
        info = {
            type:{
                'woe':math.log(float(flg1_dist)/flg0_dist),
                'iv':(flg1_dist-flg0_dist)*math.log(float(flg1_dist)/flg0_dist)
                }
            }
        if col not in self.result:
            self.result[col] = info
        else:
            self.result[col].update(info)
    
    def agg_iv(self):
        iv_dict={}
        
        for col,cats in self.result.items():
            iv_dict[col] = 0
            for cat,value in cats.items():
                iv_dict[col] += value['iv']
        
        return iv_dict
    
    
        
        
        
        
        
    
    
    
                
                
                
            


