from DDPMiner import DDPMine
from fptree import FPTree
from fptree.FPTree import *
from TransactionDatabase import TransactionDatabase
from optparse import OptionParser
import time
import codecs

# def run_ddpmine():
#     # Just some placeholder data
#     miner = DDPMine(debug=False)
#     miner.mine()

if __name__ == "__main__":

    
    database = TransactionDatabase.loadFromFile("./data/train_adt.csv",['97'],100)
    data = TransactionDatabase.loadFromFile("./data/train_adt.csv",['97'],1)
    data1 = TransactionDatabase.loadFromFile("./data/test_adt.csv",['97'],1)
    # database.cleanAndPrune(2)
    # print ("Cleaned database:")
    # for transaction in database.transactions:
    #     print(str(transaction.label))
    # print ("\nItems in FP tree and corresponding nodes:")
    tree = FPTree()
    for t in database:
        tree.add(t)

    # print(str(tree))
    miner = DDPMine(debug=True)
    start = time.clock()
    Pt = miner.mine(database,100)
    elapsed = time.clock() - start
    print("Time Total:%f"%elapsed)
    print(Pt)
    for row in Pt:
        print("Pattern:%s  label:%s"%(row[0],row[1]))
    
    
    
    
    for row in Pt:
        lb1 = 0
        lb2 = 0
        for transaction in data.transactions:
            if set(row[0]).issubset(set(transaction.itemset)):
                
                if transaction.label=="96":
                    lb1 +=1
                if transaction.label=="97":
                    lb2 +=1
        if lb1>=lb2:
            row[1] = "96"
        if lb1 < lb2:
            row[1] = "97"
    
    bingo1=0
    count1=0
    

    for row in Pt:
        for transaction in data.transactions:
            if set(row[0]).issubset(set(transaction.itemset)):
                #print("pred:",row[1],"label:",transaction)
                count1+=1
                if row[1]==transaction.label:
                    bingo1+=1
    accuracy1 = float(bingo1)/count1
    print("#############################################The accuracy of train is:%f"%(accuracy1))
    
    
    bingo2 = 0
    count2 = 0
    
    for row in Pt:
        for transaction in data1.transactions:
            if set(row[0]).issubset(set(transaction.itemset)):
                #print("pred:",row[1],"label:",transaction)
                count2+=1
                if row[1]==transaction.label:
                    bingo2+=1
    accuracy2 = float(bingo2)/count2
    print("Bingo:",bingo2,"Count",count2)
    print("##############################################The accuracy of test is:%f"%(accuracy2))
    # for transaction in data.transactions:
    #     print(str(transaction.label))


    # run_ddpmine()
