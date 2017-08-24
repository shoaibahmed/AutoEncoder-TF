import os
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.python.platform import gfile
from optparse import OptionParser
import datetime as dt
import cPickle as pkl
import time

from autoencoder import *

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
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="/netscratch/siddiqui/Bosch/data/faster-rcnn/schulze/train_cam3_cam4.idl", help="IDL file name for training")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="/netscratch/siddiqui/Bosch/data/faster-rcnn/schulze/train_cam3_cam4.idl", help="IDL file name for testing")
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=640, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=480, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=1, help="Number of channels in image for feeding into the network")
parser.add_option("--sequentialFetch", action="store_true", dest="sequentialFetch", default=False, help="Sequentially fetch images for each batch")
parser.add_option("--randomFetchTest", action="store_true", dest="randomFetchTest", default=False, help="Randomly fetch images for each test batch")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=5, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=500, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=100000, help="Progress evaluation step")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="/netscratch/siddiqui/Bosch/src/logs", help="Directory for saving logs")
parser.add_option("--modelDir", action="store", type="string", dest="modelDir", default="/netscratch/siddiqui/Bosch/src/bosch-autoencoder/", help="Directory for saving the model")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="bosch-autoencoder", help="Name to be used for saving the model")
parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="/netscratch/siddiqui/Bosch/src/outputImages", help="Directory for saving output images")

# Network Params
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=2, help="Number of classes")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

# Import custom data
import inputReader as reader
inputReader = reader.InputReader(options)

if options.trainModel:
	with tf.variable_scope('Model'):
		if options.convolutionalAutoencoder:
			# Data placeholders
			inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, options.imageHeight,
												options.imageWidth, options.imageChannels], name="inputBatchImages")
			# Create model
			# ae = autoencoder(inputBatchImages)
			ae = autoencoder_complete(inputBatchImages)
		else:
			# Data placeholders
			inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, options.imageHeight *
												options.imageWidth * options.imageChannels], name="inputBatchImages")
			# Create model
			ae = dense_autoencoder(inputBatchImages)
		
		variables_to_restore = slim.get_variables_to_restore()

	# Create list of vars to restore before train op
	# variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"])

	with tf.name_scope('Loss'):
		# Define loss
		# Reversed from slim.losses.softmax_cross_entropy(logits, labels) => tf.losses.softmax_cross_entropy(labels, logits)
	    # cost function measures pixel-wise difference
		mse_loss = tf.reduce_mean(tf.square(ae['y'] - inputBatchImages))
		# loss = mse_loss

		tf.add_to_collection('losses', mse_loss)
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
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

	# Initializing the variables
	init = tf.global_variables_initializer() # TensorFlow v0.11
	# init_local = tf.local_variables_initializer()

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

# Train model
if options.trainModel:
	with tf.Session() as sess:
		print ("Initializing model")
		# Initialize all variables
		sess.run(init)
		# sess.run(init_local)

		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.modelDir)
			os.system("rm -rf " + options.imagesOutputDirectory)
			os.system("mkdir " + options.modelDir)
			os.system("mkdir " + options.imagesOutputDirectory)

		# Restore checkpoint
		else:
			print ("Restoring from checkpoint")
			saver = tf.train.Saver(variables_to_restore)
			# saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
			saver.restore(sess, options.modelDir + options.modelName)

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

		print ("Starting network training")

		# Keep training until reach max iterations
		try:
			while True:
				start_time = time.time()
				batchImagesTrain = inputReader.getTrainBatch()
				# print ("Batch images shape: %s" % (batchImagesTrain.shape))

				# If training iterations completed
				if batchImagesTrain is None:
					print ("Training completed")
					break

				# Run optimization op (backprop)
				if options.tensorboardVisualization:
					[trainLoss, _, summary] = sess.run([loss, applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain})
					# Write logs at every iteration
					summaryWriter.add_summary(summary, step)
				else:
					[trainLoss, _] = sess.run([loss, applyGradients], feed_dict={inputBatchImages: batchImagesTrain})

				duration = time.time() - start_time
				print ("Iteration: %d, Minibatch Loss: %f (Time: %.3f sec)" % (step, trainLoss, duration))
				step += 1

				if step % options.saveStep == 0:
					# Save model weights to disk
					saver.save(sess, options.modelDir + options.modelName)
					print ("Model saved: %s" % (options.modelDir + options.modelName))

				#Check the accuracy on test data
				if step % options.evaluateStep == 0:
					# Report loss on test data
					batchImagesTest, batchLabelsTest = inputReader.getTestBatch()

					[testLoss, testAcc] = sess.run([loss, accuracy], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
					print ("Test loss: %f, Test Accuracy: %f" % (testLoss, testAcc))

					# #Check the accuracy on test data
					# if step % options.saveStepBest == 0:
					# 	# Report loss on test data
					# 	batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
					# 	[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
					# 	print ("Test loss: %f" % testLoss)

					# 	# If its the best loss achieved so far, save the model
					# 	if testLoss < bestLoss:
					# 		bestLoss = testLoss
					# 		# bestModelSaver.save(sess, best_checkpoint_dir + 'checkpoint.data')
					# 		bestModelSaver.save(sess, checkpointPrefix, global_step=0, latest_filename=checkpointStateName)
					# 		print ("Best model saved in file: %s" % checkpointPrefix)
					# 	else:
					# 		print ("Previous best accuracy: %f" % bestLoss)

		except KeyboardInterrupt:
			print("Process interrupted by user") # Save the model and exit

		# Save final model weights to disk
		saver.save(sess, options.modelDir + options.modelName)
		print ("Model saved: %s" % (options.modelDir + options.modelName))

		# Report loss on test data
		batchImagesTest = inputReader.getTestBatch()
		testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest})
		print ("Test loss (current): %f" % testLoss)

		print ("Optimization Finished!")

# Test model
if options.testModel:
	print ("Testing saved model")

	os.system("rm -rf " + options.imagesOutputDirectory)
	os.system("mkdir " + options.imagesOutputDirectory)

	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session() as session:
		saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
		saver.restore(session, options.modelDir + options.modelName)

		# nodeNames = [n.name for n in tf.get_default_graph().as_graph_def().node if n.name.startswith("Model")]
		# print(nodeNames)

		# Get reference to placeholders
		outputNode = session.graph.get_tensor_by_name("Model/output:0")
		lossNode = session.graph.get_tensor_by_name("Loss/total_loss:0")
		inputBatchImages = session.graph.get_tensor_by_name("Model/inputBatchImages:0")

		print ("Computing mse scores")
		prevBatchSize = options.batchSize
		options.batchSize = 1
		inputReaderNew = reader.InputReader(options)

		mse = []
		fileNames = []
		inputReaderNew.resetTrainBatchIndex()
		inputReaderNew.resetTestBatchIndex()
		numBatches = 0
		isTrain = False#True
		while True:
			if isTrain:
				batchImages = inputReaderNew.getTrainBatchSequential()
			else:
				batchImages = inputReaderNew.getTestBatch()
				
			if batchImages is None:
				if isTrain:
					isTrain = False
					continue
				else:
					break

			batchImagesName = inputReaderNew.getFileNames(isTrain = isTrain)
		
			loss = session.run(lossNode, feed_dict={inputBatchImages: batchImages})
			# print ('Batch loss: %.2f' % (loss))

			# for i in range(len(loss)):
			# 	mse.append(loss[i])
			# 	fileNames.append(batchImagesName[i])
			mse.append(loss)
			fileNames.append(batchImagesName[0])

			numBatches += 1

		with open('results.npy', 'wb') as fp:
			pkl.dump(fileNames, fp)
			pkl.dump(mse, fp)

		with open('results.npy', 'r') as fp:
			fileNames = pkl.load(fp)
			mse = pkl.load(fp)

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
			print ("Computing output images")
			options.batchSize = prevBatchSize
			inputReader.resetTestBatchIndex()
			numBatches = 0
			imageIndex = 0
			while True:
				batchImagesTest = inputReader.getTestBatch()
				if batchImagesTest is None:
					break
			
				[outputImages, loss] = session.run([outputNode, lossNode], feed_dict={inputBatchImages: batchImagesTest})
				print ('Batch loss: %f' % (loss))
			
				numBatches += 1

				# plot_image = np.concatenate((img_A, img_B), axis=1)
				plotImages = []
				plotTitles = []
				for i in range(batchImagesTest.shape[0]):
					squaredError = np.square(batchImagesTest[i, :, :, :] - outputImages[i, :, :, :])
					minVal = np.min(squaredError)
					maxVal = np.max(squaredError)
					# squaredError = (squaredError - minVal) / (maxVal - minVal)
					squaredError = (squaredError + minVal) / (maxVal - minVal)
					squaredError = squaredError * 255.0
					squaredError = np.uint8(squaredError)

					minVal = np.min(batchImagesTest[i, :, :, :])
					maxVal = np.max(batchImagesTest[i, :, :, :])
					# originalImage = (batchImagesTest[i, :, :, :] - minVal) / (maxVal - minVal)
					originalImage = (batchImagesTest[i, :, :, :] + minVal) / (maxVal - minVal)
					originalImage = originalImage * 255.0
					originalImage = np.uint8(originalImage)

					minVal = np.min(outputImages[i, :, :, :])
					maxVal = np.max(outputImages[i, :, :, :])
					# reconstructedImage = (outputImages[i, :, :, :] - minVal) / (maxVal - minVal)
					reconstructedImage = (outputImages[i, :, :, :] + minVal) / (maxVal - minVal)
					reconstructedImage = reconstructedImage * 255.0
					reconstructedImage = np.uint8(reconstructedImage)

					plotImages.append(np.squeeze(np.concatenate([originalImage, reconstructedImage, squaredError], axis=1)))
					plotTitles.append('mse_' + str(mse[imageIndex]) + '_rank_' + str(sortedIndices[imageIndex]))
					imageIndex += 1

				# Save image results
				# inputReader.saveLastBatchResults(outputImages, isTrain=False)
				plotImages = np.array(plotImages)
				inputReader.saveLastBatchResults(plotImages, titles=plotTitles, isTrain=False, rescale=False)


	print ("Model tested")
