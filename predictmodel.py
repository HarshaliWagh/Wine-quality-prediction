import sys

from pyspark import SparkContext, SparkConf,SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession

#Creating Spark Context
conf = SparkConf()
sc = SparkContext(conf=conf)
spark = SparkSession.builder.appName('WineApp').getOrCreate()

#Using Random Forest Model
randomForestModel = RandomForestModel.load(sc, "s3://cs643-wine/hawModel")
#randomForestModel = RandomForestModel.load(sc, "hawModel")
fpathread = sc.textFile(sys.argv[1])
# def parsePoint(line):
#     values = [float(x) for x in line.split(';')]
#     return LabeledPoint(values[11], values[0:10])
def read_csv(fpathread):
    return spark.read.format("com.databricks.spark.csv").csv(fpathread, header=True, sep=";")
#vdf = spark.read.format('com.databricks.spark.csv').csv('ValidationDataset.csv', header=True, sep=";")
vdf = read_csv(fpathread)
parsedTestData = vdf.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = randomForestModel.predict(parsedTestData.map(lambda x: x.features))
labels_Predictions = parsedTestData.map(lambda lp: lp.label).zip(predictions)

#Calculating F1 Score
metrics = MulticlassMetrics(labels_Predictions)
f1Score = metrics.weightedFMeasure()

#F1 Score
print ("F1 score on Test data = ", f1Score)
