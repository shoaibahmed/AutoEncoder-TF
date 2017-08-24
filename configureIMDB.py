import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from optparse import OptionParser

NONE = 0
TRAIN = 1
TEST = 2

def traverseDirectory(options):
	imagesFileTrain = open(options.imagesTrainOutputFile, 'w')
	imagesFileTest = open(options.imagesTestOutputFile, 'w')
	dirs = [os.path.join(options.dir, "CorrectImages"), os.path.join(options.dir, "IncorrectImages")]

	print ('Processing images')
	for currentDir in dirs:
		for root, dirs, files in os.walk(currentDir):
			print ("Directory: %s" % os.path.basename(root))

			imageType = NONE
			if (os.path.basename(root) == "CorrectImages"):
				imageType = TRAIN
			elif (os.path.basename(root) == "IncorrectImages"):
				imageType = TEST
			
			if imageType != NONE:
				print ("Processing directory: %s" % os.path.basename(root))
				for file in files:
					if file.endswith(options.searchString):
						fileName = str(os.path.abspath(os.path.join(root, file))).encode('string-escape')
						if imageType == TRAIN: # CorrectImages
							if random.random() <= 0.8:
								imagesFileTrain.write(fileName + '\n')
							else:
								imagesFileTest.write(fileName + '\n')
						elif imageType == TEST: # IncorrectImages
							imagesFileTest.write(fileName + '\n')

	imagesFileTrain.close()
	imagesFileTest.close()

if __name__ == "__main__":

	# Command line options
	parser = OptionParser()
	parser.add_option("-d", "--dir", action="store", type="string", dest="dir", default=u".", help="Root directory of training data to be searched")
	parser.add_option("--searchString", action="store", type="string", dest="searchString", default=".jpg", help="Criteria for finding relevant files")
	parser.add_option("--imagesTrainOutputFile", action="store", type="string", dest="imagesTrainOutputFile", default="train.idl", help="Name of train images file")
	parser.add_option("--imagesTestOutputFile", action="store", type="string", dest="imagesTestOutputFile", default="test.idl", help="Name of test images file")

	# Parse command line options
	(options, args) = parser.parse_args()

	traverseDirectory(options)

	print ("Done")
