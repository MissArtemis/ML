from PatternLearn.FPTree import *
#########################################################
# We want to use FPTree to Mine frequent rules--Fpgrowth
#### There are three basic steps to extract the frequent itemsets from the FPTree
# 1.Get conditional pattern bases from the FPTree
# 2.From the conditional pattern base, construct a conditional FPTree
# 3.Recursively repeat steps 1 and 2 until the tree contains a single item
##########################################################################

# ascendTree() that ascends the tree collecting the name of items it encounters

def ascendTree(leafNode,prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

# The findPrefixPath() iterates through the link list until it hits the end
# For each item it encounters, it calls ascendTree()

def findPrefixPath(treeNode):
    conditionalPattern = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            conditionalPattern[frozenset(prefixPath[1:])]=treeNode.freq
        treeNode = treeNode.link
    return conditionalPattern

if __name__ == "__main__":
    # Test
    # Test Case
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['r', 'z', 'h', 'j', 'p'],
               ['r', 'z', 'h', 'j', 'p'],
               ['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

    # header = CreateTree(simpDat,1)
    initSet = createInitSet(simpDat)
    print(initSet)
    # print(simpDat)
    # print(header)
    Fptree, header = CreateTree(initSet, 3)
    Fptree.display()
    print(findPrefixPath(header['r'][1]))


