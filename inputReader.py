import os
import os.path
import numpy as np
import skimage
import skimage.io
import skimage.transform
import cv2

class InputReader:
	def __init__(self, options):
		self.options = options

		# Read path of images together
		self.imageList = self.readImageNames(options.trainFileName)
		self.imageListTest = self.readImageNames(self.options.testFileName)
		
		# Shuffle the image list if sequential sampling is selected
		if self.options.sequentialFetch:
			np.random.shuffle(self.imageList)

		# meanFile = "meanImg.npy"
		# if os.path.isfile(meanFile):
		# 	self.meanImg = np.load(meanFile)
		# 	print ("Image mean: %.3f" % (self.meanImg))
		# else:
		# 	print ("Computing mean image from training dataset")
		# 	# Compute mean image
		# 	meanImg = np.zeros([options.imageHeight, options.imageWidth])
		# 	imagesProcessed = 0
		# 	for i in range(len(self.imageList)):
		# 		try:
		# 			img = skimage.io.imread(self.imageList[i])
		# 		except:
		# 			print ("Unable to load image: %s" % self.imageList[i])
		# 			continue
				
		# 		# Perform image scaling [0, 1]
		# 		img = img.astype(float)
		# 		img = img / 255.0

		# 		meanImg += img
		# 		imagesProcessed += 1

		# 	meanImg = meanImg / imagesProcessed
		# 	self.meanImg = meanImg
		# 	np.save("fullImageMean.npy", self.meanImg)

		# 	# Convert mean to per channel mean (single channel images)
		# 	self.meanImg = np.mean(meanImg)
		# 	np.save(meanFile, self.meanImg)

		# 	print ("Mean image computed")
		# 	# print (self.meanImg.shape)

		self.currentIndex = 0
		self.currentIndexTest = 0
		self.totalEpochs = 0
		self.totalImages = len(self.imageList)
		self.totalImagesTest = len(self.imageListTest)

		self.useEdgeImages = False

		# self.imgShape = [self.options.imageHeight, self.options.imageWidth, self.options.imageChannels]

	def readImageNames(self, imageListFile):
		"""Reads a .txt file containing image paths
		Args:
		   imageListFile: a .txt file with one /path/to/image per line
		Returns:
		   List with all fileNames in file imageListFile
		"""
		f = open(imageListFile, 'r')
		fileNames = []
		labels = []
		for line in f:
			fileName = line.strip()
			fileNames.append(fileName)

		return fileNames

	def readImagesFromDisk(self, fileNames):
		"""Consumes a list of filenames and returns images
		Args:
		  fileNames: List of image files
		Returns:
		  4-D numpy array: The input images
		"""
		images = []
		for i in range(0, len(fileNames)):			
			if self.options.verbose > 1:
				print ("Image: %s" % fileNames[i])

			# Read image
			img = skimage.io.imread(fileNames[i])

			if self.useEdgeImages:
				img = cv2.GaussianBlur(img, (3, 3), 0)
				img = cv2.Laplacian(img, cv2.CV_8U)
				# img = cv2.Canny(img, 200, 250)

				img = img.astype(float)
				# img = img / 255.0

				# img = img - 0.5

				mean = np.mean(img)
				stddev = np.std(img)
				img = (img - mean) / stddev
				# minVal = np.min(img)
				# maxVal = np.max(img)
				# scaledIm = (img - minVal) / (maxVal - minVal) # 0 to 1
				# scaledIm = (scaledIm - 0.5) / 10
				# scaledIm = (img - mean) / (maxVal - minVal) # -0.5 to 0.5
				# scaledIm = scaledIm * 10

				# Perform image scaling [-0.5, 0.5]
				# minVal = np.min(img)
				# maxVal = np.max(img)
				# interval = maxVal - minVal
				# # img = (img - minVal - (interval / 2.0)) / interval # [-0.5, 0.5]
				# img = (img - minVal) / interval # [-0.5, 0.5]
			else:
				# Perform image scaling [0, 1]
				img = img.astype(float)
				# img = img / 255.0

				# Perform mean normalization
				# img = img - self.meanImg
				mean = np.mean(img)
				stddev = np.std(img)
				img = (img - mean) / stddev

			# Extend the dimension of the image
			img = np.expand_dims(img, -1)

			# Flatten the image if shallow autoencoder is used
			if not self.options.convolutionalAutoencoder:
				img = img.flatten()
			
			# if img.shape != self.imgShape:
			# 	img = skimage.transform.resize(img, self.imgShape, preserve_range=True)
			images.append(img)

		# Convert list to ndarray
		images = np.array(images, dtype=np.float32)

		return images

	def getTrainBatch(self):
		"""Returns training images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: training images and labels in batch.
		"""
		if self.totalEpochs >= self.options.trainingEpochs:
			return None

		endIndex = self.currentIndex + self.options.batchSize
		if self.options.sequentialFetch:
			# Fetch the next sequence of images
			self.indices = np.arange(self.currentIndex, endIndex)

			if endIndex > self.totalImages:
				# Replace the indices which overshot with 0
				self.indices[self.indices >= self.totalImages] = np.arange(0, np.sum(self.indices >= self.totalImages))
		else:
			# Randomly fetch any images
			self.indices = np.random.choice(self.totalImages, self.options.batchSize)

		imagesBatch = self.readImagesFromDisk([self.imageList[index] for index in self.indices])

		self.currentIndex = endIndex
		if self.currentIndex > self.totalImages:
			print ("Training epochs completed: %f" % (self.totalEpochs + (float(self.currentIndex) / self.totalImages)))
			self.currentIndex = self.currentIndex - self.totalImages
			self.totalEpochs = self.totalEpochs + 1

			# Shuffle the image list if not random sampling at each stage
			if self.options.sequentialFetch:
				np.random.shuffle(self.imageList)

		return imagesBatch

	def getFileNames(self, isTrain=True):
		if isTrain:
			imageList = [self.imageList[index] for index in self.indices]
		else:
			imageList = [self.imageListTest[index] for index in self.indices]

		return imageList

	def resetTrainBatchIndex(self):
		self.currentIndex = 0
		self.totalEpochs = 0

	def resetTestBatchIndex(self):
		self.currentIndexTest = 0

	def getTrainBatchSequential(self):
		"""Returns training images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: training images and labels in batch.
		"""
		# Optional Image and Label Batching
		if self.currentIndex >= self.totalImages:
			return None

		endIndex = self.currentIndex + self.options.batchSize
		if endIndex > self.totalImages:
			endIndex = self.totalImages
		self.indices = np.arange(self.currentIndex, endIndex)
		imagesBatch = self.readImagesFromDisk([self.imageList[index] for index in self.indices])
		self.currentIndex = endIndex

		return imagesBatch

	def getTestBatch(self):
		"""Returns testing images and labels in batch
		Args:
		  None
		Returns:
		  Two numpy arrays: test images and labels in batch.
		"""
		# Optional Image and Label Batching
		if self.currentIndexTest >= self.totalImagesTest:
			return None

		if self.options.randomFetchTest:
			self.indices = np.random.choice(self.totalImagesTest, self.options.batchSize)
			imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices])
			
		else:
			endIndex = self.currentIndexTest + self.options.batchSize
			if endIndex > self.totalImagesTest:
				endIndex = self.totalImagesTest
			self.indices = np.arange(self.currentIndexTest, endIndex)
			imagesBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices])
			self.currentIndexTest = endIndex

		return imagesBatch

	def saveLastBatchResults(self, outputImages, titles, isTrain=True, rescale=True):
		"""Saves the results of last retrieved image batch
		Args:
		  outputImages: 4D Numpy array [batchSize, H, W, numClasses]
		  isTrain: If the last batch was training batch
		Returns:
		  None
		"""
		if isTrain:
			imageNames = [self.imageList[index] for index in self.indices]
		else:
			imageNames = [self.imageListTest[index] for index in self.indices]

		# outputImages = np.squeeze(outputImages)
		# print(outputImages.shape)
		# Iterate over each image name and save the results
		for i in range(self.indices.shape[0]):
			imageName = imageNames[i].split('/')
			imageName = imageName[-1]

			if isTrain:
				imageName = self.options.imagesOutputDirectory + '/' + 'train_' + imageName[:-4] + '_' + titles[i] + imageName[-4:]
			else:
				imageName = self.options.imagesOutputDirectory + '/' + 'test_' + imageName[:-4] + '_' + titles[i] + imageName[-4:]
			# print(imageName)

			# Save foreground probability
			# im = ((outputImages[i, :, :] + self.meanImg) * 255)
			# im = im.astype(np.uint8)	# Convert image from float to unit8 for saving
			im = outputImages[i, :, :]

			if rescale:
				minVal = np.min(im)
				maxVal = np.max(im)
				scaledIm = (im - minVal) / (maxVal - minVal)
				# A_scaled.convertTo(display, CV_8UC1, 255.0, 0); 
				scaledIm = scaledIm * 255.0
				scaledIm = np.uint8(scaledIm)

				skimage.io.imsave(imageName, scaledIm)
			else:
				if "niO" in imageNames[i]:
					titles[i] += " niO"
				else:
					titles[i] += " iO"

				# Write some Text
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(im, titles[i], (10, im.shape[0] - 10), font, 1, (255,255,255), 2)

				skimage.io.imsave(imageName, im)
