# Define the structure of the FPNode
class FPNode:
    def __init__(self, name, freq, parent):
        self.name = name
        self.freq = freq
        self.child = {}  # An empty dictionary for the children in the node
        self.link = None  # link used to link similar items
        self.parent = parent

    def add_count(self, freq):  # Add freq at a given num
        self.freq += freq

    def display(self, ind=1):
        print(' ' * ind, self.name, ' ', self.freq)
        for child in self.child.values():
            child.display(ind + 1)

# Test Case

# root = FPNode('Human', 9, None)
# root.child['eye'] = FPNode('eye', 2, root)
# root.child['mouth'] = FPNode('mouth',1,root)
# mouth = FPNode('mouth',1,root)
# mouth.child['teeth'] = FPNode('teeth',32,mouth)
# root.display()
# mouth.display()
####################################################################################

# Construct FPTree
##################
# Create FPTree from dataset
def CreateTree(dataset,minSup=1): #Default set Min Support as 1
    headerTable = {}
    # We need travel through the whole dataset twice
    for trans in dataset:# First pass counts frequency of each item
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + 1
      # store the freq dictionary
    # return headerTable
    for k in list(headerTable): # Remove item which does not reach the min Support
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys()) # Collect all items that are larger than min Support
    print("freqItemKeys:",freqItemSet)

    # If there are no item meets the requirement
    if len(freqItemSet)==0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k],None] # Reformat headerTable to use link
    print("headerTable: ",headerTable)
    Tree = FPNode('Null', 1, None) # Create an empty Tree
    # Travel through the dataset for the second time
    for tranSet,freq in dataset.items():
        local_id = {}
        for item in tranSet: # Put items in order
            if item in freqItemSet:
                local_id[item] = headerTable[item][0]
        if len(local_id)>0:
            orderedItems = [x[0] for x in sorted(local_id.items(), key=lambda y:y[1], reverse = True)]
            print(orderedItems)
            updateTree(orderedItems,Tree,headerTable,freq) # Build the tree with frequent items
    return Tree,headerTable #Return the tree and headerTable

# Use updateTree grow FPTree
def updateTree(items, Tree, headerTable, freq):
    if items[0] in Tree.child: # Check whether item is already in the tree
        Tree.child[items[0]].add_count(freq)  # if item already in the tree, add the count to the node
    else:
        Tree.child[items[0]] = FPNode(items[0], freq, Tree)  # if it is the first time the item occurs, add this item as a new node
        if headerTable[items[0]][1] == None:  # If there is no link
            headerTable[items[0]][1] = Tree.child[items[0]]
        else:
            updateHeader(headerTable[items[0]][1],Tree.child[items[0]])
    if len(items)>1: # If there are stil items remain
        updateTree(items[1::],Tree.child[items[0]],headerTable,freq)
def updateHeader(Start,End):
    while(Start.link != None):
        Start = Start.link
    Start.link = End

# To take the Input Data as a dictionary
def createInitSet(dataset):
    retDict = {}
    trans_dic  = {}
    for trans in dataset:
        retDict[frozenset(trans)] = 0
    for trans in dataset:
        retDict[frozenset(trans)] += 1
    return retDict



#Test Case
# simpDat = [['r', 'z', 'h', 'j', 'p'],
#                ['r', 'z', 'h', 'j', 'p'],
#                ['r', 'z', 'h', 'j', 'p'],
#                ['r', 'z', 'h', 'j', 'p'],
#                ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
#                ['z'],
#                ['r', 'x', 'n', 'o', 's'],
#                ['y', 'r', 'x', 'z', 'q', 't', 'p'],
#                ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
#
# # header = CreateTree(simpDat,1)
# initSet = createInitSet(simpDat)
# print(initSet)
# print(simpDat)
# print(header)
# Fptree,header = CreateTree(initSet,3)
# Fptree.display()

##############################################################
