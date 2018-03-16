import sys
from pyspark import SparkContext


def chargement(sc,dir):
	rdd_image=sc.binaryFiles(dir)
	return rdd_image
	
	
def main():
	sc= SparkContext(master="local[2]",appName="2typesDogs")
	rdd_image=chargement(sc,sys.argv[1])
	print rdd_image.first()

if __name__ == "__main__":
	main()
	