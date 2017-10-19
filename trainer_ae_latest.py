import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.platform import gfile
from optparse import OptionParser
import datetime as dt
# import cPickle as pkl
import pickle as pkl
import time

from autoencoder import *
import cv2

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-o", "--computeOutputImages", action="store_true", dest="computeOutputImages", default=False, help="Compute reconstructed images for autoencoder")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")
parser.add_option("--convolutionalAutoencoder", action="store_true", dest="convolutionalAutoencoder", default=False, help="Use convolutional autoencoder instead of shallow autoencoder")

# Input Reader Params
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="./io.txt", help="Text file name to be used for training")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="./nio.txt", help="Text file name to be used for testing")
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=640, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=480, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=1, help="Number of channels in image for feeding into the network")
parser.add_option("--sequentialFetch", action="store_true", dest="sequentialFetch", default=False, help="Sequentially fetch images for each batch")
parser.add_option("--randomFetchTest", action="store_true", dest="randomFetchTest", default=False, help="Randomly fetch images for each test batch")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=50, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=3, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=100000, help="Progress evaluation step")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs/", help="Directory for saving logs")
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="./model/", help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="bosch-autoencoder", help="Name to be used for saving the model")
parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="/netscratch/siddiqui/Bosch/AutoEnc-output/", help="Directory for saving output images")

# Network Params
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=2, help="Number of classes")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Reads an image from a file, decodes it into a dense tensor
def _parse_function(filename):
	image_string = tf.read_file(filename)
	img = tf.image.decode_image(image_string)

	img = tf.reshape(img, [options.imageHeight, options.imageWidth, options.imageChannels])
	# if options.convolutionalAutoencoder:
	# 	img = tf.reshape(img, [options.imageHeight, options.imageWidth, options.imageChannels])
	# else:
	# 	img = tf.reshape(img, [options.imageHeight * options.imageWidth * options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor
	
	# Normalize the image
	img = tf.image.per_image_standardization(img)

	return img, filename

# A vector of filenames
dataFileName = options.trainFileName if options.trainModel else options.testFileName
print ("Loading data from file: %s" % (dataFileName))
with open(dataFileName) as f:
	imageFileNames = f.readlines()
	imageFileNames = [x.strip() for x in imageFileNames]
	filenames = tf.constant(imageFileNames)

dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(_parse_function)
dataset = dataset.batch(options.batchSize if options.trainModel else 1)

# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()

with tf.variable_scope('Model'):
	inputBatchImages, inputBatchImageNames = iterator.get_next()
	print ("Data shape: %s" % str(inputBatchImages.get_shape()))
	
	# Create model
	if options.convolutionalAutoencoder:
		# ae = auto_encoder_with_spatial_transformer(inputBatchImages, options.trainModel)
		ae = convolutional_auto_encoder(inputBatchImages, options.trainModel)
	else:
		ae = dense_autoencoder(inputBatchImages, options.trainModel)
	
	variables_to_restore = slim.get_variables_to_restore()

with tf.name_scope('Loss'):
	# Define loss
	# Reversed from slim.losses.softmax_cross_entropy(logits, labels) => tf.losses.softmax_cross_entropy(labels, logits)
	# cost function measures pixel-wise difference
	mse_loss = tf.reduce_mean(tf.square(ae['y'] - inputBatchImages))
	tf.add_to_collection('losses', mse_loss)

	# Add L2 regularization on the feature vector
	l2Lambda = 1e-8
	l2RegLoss = l2Lambda * tf.reduce_sum(tf.square(ae['z']))
	tf.add_to_collection('losses', l2RegLoss)
	
	loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

# with tf.name_scope('Accuracy'):
# 	correct_predictions = tf.equal(tf.argmax(end_points['Predictions'], 1), tf.argmax(inputBatchLabels, 1))
# 	accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

with tf.name_scope('Optimizer'):
	# Define Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

	# Op to calculate every variable gradient
	gradients = tf.gradients(loss, tf.trainable_variables())
	gradients = list(zip(gradients, tf.trainable_variables()))

	# Op to update all variables according to their gradient
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

# Initializing the variables
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

if options.tensorboardVisualization:
	# Create a summary to monitor cost tensor
	tf.summary.scalar("loss", loss)

	# Create summaries to visualize weights
	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)
	# Summarize all gradients
	for grad, var in gradients:
		tf.summary.histogram(var.name + '/gradient', grad)

	# Merge all summaries into a single op
	mergedSummaryOp = tf.summary.merge_all()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

bestLoss = 1e9
step = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Train model
if options.trainModel:
	# with tf.train.MonitoredTrainingSession() as sess:
	with tf.Session(config=config) as sess:
		print ("Initializing model")
		# Initialize all variables
		sess.run(init)
		sess.run(init_local)

		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.modelDir)
			os.system("mkdir " + options.modelDir)

		# Restore checkpoint
		else:
			print ("Restoring from checkpoint")
			saver = tf.train.Saver(variables_to_restore)
			saver.restore(sess, options.modelDir + options.modelName)

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

		print ("Starting network training")

		# Keep training until reach max iterations
		for epoch in range(options.trainingEpochs):
			sess.run(iterator.initializer)
			try:
				print ("Starting %d Epoch" % (epoch+1))
				while True:
					start_time = time.time()
					# Run optimization op (backprop)
					if options.tensorboardVisualization:
						[trainLoss, _, summary] = sess.run([loss, applyGradients, mergedSummaryOp])
						# Write logs at every iteration
						summaryWriter.add_summary(summary, step)
					else:
						[trainLoss, _] = sess.run([loss, applyGradients])

					duration = time.time() - start_time
					print ("Iteration: %d, Minibatch Loss: %f (Time: %.3f sec)" % (step, trainLoss, duration))
					step += 1

					if step % options.saveStep == 0:
						# Save model weights to disk
						saver.save(sess, options.modelDir + options.modelName)
						print ("Model saved: %s" % (options.modelDir + options.modelName))
				
			except KeyboardInterrupt:
				print("Process interrupted by user") # Save the model and exit
				break

			except tf.errors.OutOfRangeError:
				print ("Epoch %d finished" % (epoch+1))
		
		# Save final model weights to disk
		saver.save(sess, options.modelDir + options.modelName)
		print ("Model saved: %s" % (options.modelDir + options.modelName))

		print ("Optimization Finished!")

# Test model
if options.testModel:
	print ("Testing saved model")

	os.system("rm -rf " + options.imagesOutputDirectory)
	os.system("mkdir " + options.imagesOutputDirectory)

	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session(config=config) as sess:
		# Initialize all variables
		sess.run(init)
		sess.run(init_local)

		# saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
		saver.restore(sess, options.modelDir + options.modelName)

		print ("Computing mse scores")
		mse = []
		fileNames = []
		
		# Keep training until reach max iterations
		sess.run(iterator.initializer)
		try:
			while True:
				imageNames, testLoss = sess.run([inputBatchImageNames, loss])
				print ("Processed image: %s with loss: %f" % (imageNames[0].decode("utf-8"), testLoss))
				mse.append(testLoss)
				fileNames.append(imageNames[0].decode("utf-8"))

		except KeyboardInterrupt:
			print("Process interrupted by user") # Save the model and exit
		except tf.errors.OutOfRangeError:
			print ("Completed testing on the test set")

		print ("Number of files: %d" % (len(fileNames)))
		assert len(fileNames) == len(mse)

		with open('results.npy', 'wb') as fp:
			pkl.dump(fileNames, fp)
			pkl.dump(mse, fp)

		# with open('results.npy', 'r') as fp:
		# 	fileNames = pkl.load(fp)
		# 	mse = pkl.load(fp)

		mse = np.array(mse)
		fileNames = np.array(fileNames)
		sortedIndices = np.argsort(mse)
		
		sortedMSE = mse[sortedIndices]
		sortedFileNames = fileNames[sortedIndices]

		with open("output.txt", "w") as file:
			for i in range(len(sortedFileNames)):
				file.write(sortedFileNames[i] + " " + str(sortedMSE[i]) + "\n")
				# file.write(sortedFileNames[i] + " " + 'mse_' + str(mse[i]) + '_rank_' + str(sortedIndices[i]) + "\n")

		if options.computeOutputImages:
			imageIndex = 0
			font = cv2.FONT_HERSHEY_SIMPLEX
			sess.run(iterator.initializer)
			print ("Computing output images")
			try:
				while True:
					[batchImagesTestNames, batchImagesTest, outputImages, testLoss] = sess.run([inputBatchImageNames, inputBatchImages, ae['y'], loss])
					# print ('Batch loss: %f' % (testLoss))

					plotImages = []
					plotTitles = []
					for i in range(batchImagesTest.shape[0]):
						squaredError = np.square(batchImagesTest[i, :, :, :] - outputImages[i, :, :, :])
						minVal = np.min(squaredError)
						maxVal = np.max(squaredError)
						# squaredError = (squaredError - minVal) / (maxVal - minVal)
						squaredError = (squaredError - minVal) / (maxVal - minVal)
						squaredError = squaredError * 255.0
						squaredError = np.uint8(squaredError)

						minVal = np.min(batchImagesTest[i, :, :, :])
						maxVal = np.max(batchImagesTest[i, :, :, :])
						# originalImage = (batchImagesTest[i, :, :, :] - minVal) / (maxVal - minVal)
						originalImage = (batchImagesTest[i, :, :, :] - minVal) / (maxVal - minVal)
						originalImage = originalImage * 255.0
						originalImage = np.uint8(originalImage)

						minVal = np.min(outputImages[i, :, :, :])
						maxVal = np.max(outputImages[i, :, :, :])
						# reconstructedImage = (outputImages[i, :, :, :] - minVal) / (maxVal - minVal)
						reconstructedImage = (outputImages[i, :, :, :] - minVal) / (maxVal - minVal)
						reconstructedImage = reconstructedImage * 255.0
						reconstructedImage = np.uint8(reconstructedImage)

						plotImage = np.squeeze(np.concatenate([originalImage, reconstructedImage, squaredError], axis=1))
						plotTitle = 'mse_' + str(mse[imageIndex]) + '_rank_' + str(sortedIndices[imageIndex])

						# Save the image
						imageName = batchImagesTestNames[i].decode("utf-8")
						imageName = imageName[imageName.rfind('/')+1:imageName.rfind('.')]
						print ("Processing image: %s" % (imageName))

						outputImagePath = os.path.join(options.imagesOutputDirectory, imageName + '_' + plotTitle + ".jpg")
						# print ("Saving output: %s" % outputImagePath)
						
						# Write some Text
						cv2.putText(plotImage, plotTitle, (10, plotImage.shape[0] - 10), font, 1, (255,255,255), 2)
						cv2.imwrite(outputImagePath, plotImage)

						# plotImages.append(np.squeeze(np.concatenate([originalImage, reconstructedImage, squaredError], axis=1)))
						# plotTitles.append('mse_' + str(mse[imageIndex]) + '_rank_' + str(sortedIndices[imageIndex]))
						imageIndex += 1

			except tf.errors.OutOfRangeError:
				print ("Computed all output images")

			# Save image results
			# plotImages = np.array(plotImages)
			# inputReader.saveLastBatchResults(plotImages, titles=plotTitles, isTrain=False, rescale=False)

	print ("Model tested")
