import sys
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext
from extract_features import extract_features_from_binary2, get_model


def chargement(sc, dir):
	rdd_image = sc.binaryFiles(dir).cache()
	return rdd_image


def split_rdd(rdd):
	"""
		Separate a rdd into two weighted rdds train(70%) and test(30%)
		:param rdd
	"""
	SPLIT_WEIGHT = 0.5
	(rdd_train, rdd_test) = rdd.randomSplit([SPLIT_WEIGHT, 1 - SPLIT_WEIGHT])
	return rdd_train, rdd_test


def prepare_data(rdd):
	return rdd.map(lambda name_feature: LabeledPoint(float(1) if "yorkshire" in name_feature[0] else float(0), name_feature[1]))



def main():
	sc = SparkContext(master="local[2]",appName="2typesDogs")
	rdd_image = chargement(sc, sys.argv[1])

	rdd_train, rdd_test = split_rdd(rdd_image)

	rdd_train = rdd_train.map(lambda name_content: (name_content[0], extract_features_from_binary2(name_content[1])))
	rdd_test = rdd_test.map(lambda name_content: (name_content[0], extract_features_from_binary2(name_content[1])))

	svm=SVMWithSGD.train(prepare_data(rdd_train))
	rdd_test = rdd_test.map(lambda name_content: (name_content[0], svm.predict(name_content[1])))
	rdd_test = rdd_test.map(lambda name_predicted: (float(1) if 'yorkshire' in name_predicted[0] else float(0), name_predicted[1]))
	nb_dogs_test = rdd_test.count()
	nb_correct_dogs = rdd_test.filter(lambda reel_predicted: reel_predicted[0]==reel_predicted[1]).count()
	accuracy = nb_correct_dogs/nb_dogs_test

	print ("Number of correct predictions :",nb_correct_dogs)
	print ("Total number of dogs :",nb_dogs_test)
	print ("Accuracy :",accuracy)



if __name__ == "__main__":
	main()
