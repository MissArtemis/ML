# import numpy as np
# a = np.array([[1,2],[3,4]])
# item_freq = { }
# p
# rint(a)
# for (a,b), x in np.ndenumerate(a):
#     print((a,b),x)
#     if x != '':
#         if x not in item_freq:
#             item_freq[x] = 1
#         else:
#             item_freq[x] += 1
# item_freq = {k:v for k,v in item_freq.items() if v >=0 }
# print(item_freq)
#
# b = [1,2,3]
# print(b[1::])

# from pyspark.ml.fpm import FPGrowth
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.appName("test").getOrCreate()
# df = spark.createDataFrame([
#     (0, [1, 2, 5]),
#     (1, [1, 2, 3, 5]),
#     (2, [1, 2]),
#     (3, [1,2,9]),
#     (4, [1,2,9]),
#     (5, [1,2,9]),
#     (6, [1,2,9]),
#     (7,[1,2,3,4,5]),
#     (8,[1,2,3,4,5,7,8])
#
# ], ["id", "items"])
#
# fpGrowth = FPGrowth(itemsCol="items", minSupport=0.4, minConfidence=0.7)
# model = fpGrowth.fit(df)
#
# # Display frequent itemsets.
# model.freqItemsets.show()
#
# # Display generated association rules.
# model.associationRules.show()
#
# # transform examines the input items against all the association rules and summarize the
# # consequents as prediction
# model.transform(df).show()
import numpy as np
a = [[1,2,3],[4,5,6]]
ah = np.mat(a)
print(ah[1])
print(ah[0][0])
print(ah[0])