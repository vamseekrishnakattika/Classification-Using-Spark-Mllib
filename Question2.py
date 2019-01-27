
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint


conf = SparkConf().setMaster("local").setAppName("Question2")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()


glass = sc.textFile("glass.data").map(lambda x: x.split(","))

feature_vector = glass.map(lambda x: LabeledPoint(float(x[10]),[float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),float(x[6]),float(x[7]),float(x[8]),float(x[9])]))

(trainingData, testData) = feature_vector.randomSplit([0.6, 0.4])

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(trainingData, numClasses=8, categoricalFeaturesInfo={},impurity='gini', maxDepth=20, maxBins=100)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
accuracy = 100.0 * labelsAndPredictions.filter(lambda lp: lp[0] == lp[1]).count() / float(testData.count())
print('Decision tree model accuracy = {}'.format(accuracy))


model = NaiveBayes.train(trainingData, 1.0)

# Make prediction and test accuracy.
predictionAndLabel = testData.map(lambda p: (model.predict(p.features), p.label))
accuracy = 100.0 * predictionAndLabel.filter(lambda pl: pl[0] == pl[1]).count() / testData.count()
print('Naive Bayes model accuracy = {}'.format(accuracy))

