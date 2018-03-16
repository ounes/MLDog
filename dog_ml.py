import sys
from pyspark import SparkContext


def chargement(sc,dir):
	rdd_image=sc.binaryFiles(dir).cache()
	return rdd_image
	
def split_rdd(rdd):
	(rdd_train, rdd_test) = rdd.randomSplit([0.7,0.3])
	return rdd_train, rdd_test
	
def main():
	sc= SparkContext(master="local[2]",appName="2typesDogs")
	rdd_image=chargement(sc,sys.argv[1])
	
	rdd_train, rdd_test = split_rdd(rdd_image)


if __name__ == "__main__":
	main()
	