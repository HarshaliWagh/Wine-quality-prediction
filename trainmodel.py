#Importing libraries for spark
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf,SQLContext
from pyspark.sql import SparkSession
#Creating spark context
conf = SparkConf()
sc = SparkContext("local",conf=conf)
spark = SparkSession.builder.appName('WineApp').getOrCreate()
# tData = sc.textFile("TrainingDataset.csv")
# header = tData.first()
# rows = tData.filter(lambda x: x != header)

df = spark.read.format('com.databricks.spark.csv').csv('s3://cs643-wine/TrainingDataset.csv', header=True, sep=";")

# def parsePoint(line):
#     # values = [float(x) for x in line.split(';')]
#     return LabeledPoint(values[11], values[0:10])

parsedTData = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

#Training data using Random Forest
model = RandomForest.trainClassifier(parsedTData, numClasses=11, categoricalFeaturesInfo={},numTrees=3,impurity='gini', maxDepth=4, maxBins=32)

# vData = sc.textFile("ValidationDataset.csv")
# header = vData.first()
# rows = vData.filter(lambda x: x != header)

vdf = spark.read.format('com.databricks.spark.csv').csv('s3://cs643-wine/ValidationDataset.csv', header=True, sep=";")

parsedVData = vdf.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(parsedVData.map(lambda x: x.features))
labels_Predictions = parsedVData.map(lambda lp: lp.label).zip(predictions)

#Calculating F1 Score on validation data
metrics = MulticlassMetrics(labels_Predictions)
f1Score = metrics.weightedFMeasure()

print ("F1 Score on validation data = ", f1Score)

#Saving model
model.save(sc, "s3://cs643-wine/hawModel")
#model.save(sc, "hawModel")
#model.write().overwrite().save("RandomForestClassifier.model")
