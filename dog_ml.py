import sys
from pyspark import SparkContext
from extract_features import extract_features_from_binary, get_model

def chargement(sc, dir):
	rdd_image = sc.binaryFiles(dir).cache()
	return rdd_image


def split_rdd(rdd):
	"""
		Separate a rdd into two weighted rdds train(70%) and test(30%)
		:param rdd
	"""
	SPLIT_WEIGHT = 0.7
	(rdd_train, rdd_test) = rdd.randomSplit([SPLIT_WEIGHT, 1 - SPLIT_WEIGHT])
	return rdd_train, rdd_test

def main():
	sc = SparkContext(master="local[2]",appName="2typesDogs")
	rdd_image = chargement(sc, sys.argv[1])

	rdd_train, rdd_test = split_rdd(rdd_image)

	print("[INFO]", len(extract_features_from_binary(get_model(), rdd_train.first()[1])))



if __name__ == "__main__":
	main()
