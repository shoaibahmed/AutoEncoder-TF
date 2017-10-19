import tensorflow as tf
import numpy as np
import math

def lrelu_func(x, leak = 0.2):
	return tf.maximum(x, leak * x)

def lrelu(x, name, leak = 0.2):
	with tf.name_scope(name):
		return tf.maximum(x, leak * x)

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
	"""
	   Unpooling layer after max_pool_with_argmax.
	   Args:
		   updates:   max pooled output tensor
		   mask:      argmax indices
		   ksize:     ksize is the same as for the pool
	   Return:
		   unpool:    unpooling tensor
	"""
	with tf.variable_scope(scope):
		input_shape = pool.get_shape().as_list()
		output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
		pool_ = tf.reshape(pool, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]])
		batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
		b = tf.ones_like(ind) * batch_range
		b = tf.reshape(b, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
		ind_ = tf.reshape(ind, [input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 1])
		ind_ = tf.concat(1, [b, ind_])
		ref = tf.Variable(tf.zeros([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]))
		ret = tf.scatter_nd_update(ref, ind_, pool_)
		ret = tf.reshape(ret, [output_shape[0], output_shape[1], output_shape[2], output_shape[3]])
		return ret

def get_deconv_filter(f_shape):
	width = f_shape[0]
	height = f_shape[1]
	f = math.ceil(width/2.0)
	c = (2 * f - 1 - f % 2) / (2.0 * f)
	bilinear = np.zeros([f_shape[0], f_shape[1]])
	for x in range(width):
		for y in range(height):
			value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
			bilinear[x, y] = value
	weights = np.zeros(f_shape)
	for i in range(f_shape[2]):
		weights[:, :, i, i] = bilinear

	init = tf.constant_initializer(value=weights,
									dtype=tf.float32)
	return tf.get_variable(name="up_filter", initializer=init,
							shape=weights.shape)

def upscore_layer(bottom, shape, num_classes, name, phase, 
				num_in_features, activation=lrelu_func,
				ksize=4, stride=2):
	strides = [1, stride, stride, 1]
	with tf.variable_scope(name):
		# in_features = bottom.get_shape()[3].value
		in_features = num_in_features

		if shape is None:
			# Compute shape out of Bottom
			in_shape = tf.shape(bottom)

			h = ((in_shape[1] - 1) * stride) + 1
			w = ((in_shape[2] - 1) * stride) + 1
			new_shape = [in_shape[0], h, w, num_classes]
		else:
			new_shape = [shape[0], shape[1], shape[2], num_classes]
		output_shape = tf.stack(new_shape)

		f_shape = [ksize, ksize, num_classes, in_features]

		# create
		num_input = ksize * ksize * in_features / stride
		stddev = (2 / num_input)**0.5

		weights = get_deconv_filter(f_shape)
		deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
										strides=strides, padding='SAME')
		if activation is not None:
			deconv = activation(deconv)

	return deconv

def convolutional_auto_encoder(x, phase):
	# Input size: 480 x 640
	bottleneckSize = 32

	# Convolutional backbone
	net = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='SAME', activation=None, name='conv1') # Output size: 240 x 320
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv1_bn')
	net = lrelu(net, name='conv1_lrelu')
	# conv1_shape = net.get_shape()
	conv1_shape = tf.shape(net)
	print ("Conv1 shape: %s" % (net.get_shape()))

	net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), strides=(2, 2), padding='SAME', activation=None, name='conv2') # Output size: 120 x 160
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv2_bn')
	net = lrelu(net, name='conv2_lrelu')
	# conv2_shape = net.get_shape()
	conv2_shape = tf.shape(net)
	print ("Conv2 shape: %s" % (net.get_shape()))

	net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv3') # Output size: 120 x 160
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv3_bn')
	net = lrelu(net, name='conv3_lrelu')
	# conv3_shape = net.get_shape()
	conv3_shape = tf.shape(net)
	net, pool3_argmax = tf.nn.max_pool_with_argmax(input=net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool3') # Output size: 60 x 80
	print ("Conv3 shape: %s" % (net.get_shape()))

	net = tf.layers.conv2d(inputs=net, filters=384, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv4') # Output size: 60 x 80
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv4_bn')
	net = lrelu(net, name='conv4_lrelu')
	# net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool4') # Output size: 30 x 40
	# conv4_shape = net.get_shape()
	conv4_shape = tf.shape(net)
	net, pool4_argmax = tf.nn.max_pool_with_argmax(input=net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool4') # Output size: 30 x 40
	print ("Conv4 shape: %s" % (net.get_shape()))

	net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv5') # Output size: 30 x 40
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv5_bn')
	net = lrelu(net, name='conv5_lrelu')
	# conv5_shape = net.get_shape()
	conv5_shape = tf.shape(net)
	# net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool5') # Output size: 15 x 20
	net, pool5_argmax = tf.nn.max_pool_with_argmax(input=net, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pool5') # Output size: 15 x 20
	print ("Conv5 shape: %s" % (net.get_shape()))

	net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(7, 7), strides=(1, 1), padding='VALID', activation=None, name='conv6') # Output size: 9 x 14
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv6_bn')
	net = lrelu(net, name='conv6_lrelu')
	# conv6_shape = net.get_shape()
	conv6_shape = tf.shape(net)
	print ("Conv6 shape: %s" % (net.get_shape()))

	# Fully-connected network
	z = tf.layers.conv2d(inputs=net, filters=bottleneckSize, kernel_size=(1, 1), strides=(1, 1), padding='SAME', activation=None, name='bottleneck') # Output size: 9 x 14 x bottleneckSize
	z = tf.layers.batch_normalization(inputs=z, center=True, scale=True, training=phase, name='bottleneck_bn')
	z = lrelu(z, name='bottleneck_lrelu')
	print ("Bottleneck (Z) shape: %s" % (z.get_shape()))

	# Decoder network
	net = tf.layers.conv2d_transpose(inputs=z, filters=128, kernel_size=(7, 7), strides=(1, 1), padding='VALID', activation=None, name='conv1_transpose') # Output size: 15 x 20
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv1_transpose_bn')
	net = lrelu(net, name='conv1_transpose_lrelu')
	print ("Conv1 transpose shape: %s" % (net.get_shape()))

	# net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv2_transpose') # Output size: 30 x 40
	# net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv2_transpose_bn')
	# net = lrelu(net, name='conv2_transpose_lrelu')

	# net = unpool(net, pool5_argmax) # Max-unpooling
	# net = upscore_layer(net, [-1, int(conv5_shape[1]), int(conv5_shape[2]), conv5_shape[3]], num_classes=32, name='Upscore_1')
	
	net = upscore_layer(net, conv5_shape, num_classes=64, num_in_features=128, name='Upscore_1', phase=phase)
	print ("Conv2 transpose shape: %s" % (net.get_shape()))

	# net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv3_transpose') # Output size: 60 x 80
	# net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv3_transpose_bn')
	# net = lrelu(net, name='conv3_transpose_lrelu')
	# net = unpool(net, pool4_argmax) # Max-unpooling

	net = upscore_layer(net, conv4_shape, num_classes=32, num_in_features=64, name='Upscore_2', phase=phase)
	print ("Conv3 transpose shape: %s" % (net.get_shape()))

	# net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv4_transpose') # Output size: 120 x 160
	# net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv4_transpose_bn')
	# net = lrelu(net, name='conv4_transpose_lrelu')
	# net = unpool(net, pool3_argmax) # Max-unpooling
	net = upscore_layer(net, conv3_shape, num_classes=16, num_in_features=32, name='Upscore_3', phase=phase)
	print ("Conv4 transpose shape: %s" % (net.get_shape()))

	# net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='SAME', activation=None, name='conv5_transpose') # Output size: 240 x 320
	# net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv5_transpose_bn')
	# net = lrelu(net, name='conv5_transpose_lrelu')
	net = upscore_layer(net, conv1_shape, num_classes=8, num_in_features=16, name='Upscore_4', phase=phase)
	print ("Conv5 transpose shape: %s" % (net.get_shape()))

	# net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=(7, 7), strides=(2, 2), padding='SAME', activation=None, name='conv6_transpose') # Output size: 480 x 640
	# net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv6_transpose_bn')
	# net = lrelu(net, name='conv6_transpose_lrelu')
	net = upscore_layer(net, tf.shape(x), num_classes=1, num_in_features=8, name='Upscore_5', phase=phase, activation=None)
	print ("Conv6 transpose shape: %s" % (net.get_shape()))

	y = tf.reshape(net, [-1, 480, 640, 1], name="output")

	tf.summary.image('Original Image', x, max_outputs=3)
	tf.summary.image('Reconstructed Image', y, max_outputs=3)

	return {'x': x, 'z': z, 'y': y}

def auto_encoder_with_spatial_transformer(x, phase):
	# Input size: 480 x 640

	# Convolutional backbone
	net = tf.layers.conv2d(inputs=x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='SAME', activation=None, name='conv1') # Output size: 240 x 320
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv1_bn')
	net = lrelu(net, name='conv1_lrelu')

	net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), strides=(2, 2), padding='SAME', activation=None, name='conv2') # Output size: 120 x 160
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv2_bn')
	net = lrelu(net, name='conv2_lrelu')

	net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv3') # Output size: 120 x 160
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv3_bn')
	net = lrelu(net, name='conv3_lrelu')
	net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool3') # Output size: 60 x 80

	net = tf.layers.conv2d(inputs=net, filters=384, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv4') # Output size: 60 x 80
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv4_bn')
	net = lrelu(net, name='conv4_lrelu')
	net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool4') # Output size: 30 x 40

	net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='SAME', activation=None, name='conv5') # Output size: 30 x 40
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv5_bn')
	net = lrelu(net, name='conv5_lrelu')
	net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2), name='pool5') # Output size: 15 x 20

	net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(7, 7), strides=(1, 1), padding='VALID', activation=None, name='conv6') # Output size: 9 x 14
	net = tf.layers.batch_normalization(inputs=net, center=True, scale=True, training=phase, name='conv6_bn')
	net = lrelu(net, name='conv6_lrelu')

	# Fully-connected network
	numberOfTemplates = 32
	z = tf.contrib.layers.flatten(net)
	z = tf.layers.dense(inputs=z, units=numberOfTemplates, activation=None, name='fc1')
	z = lrelu(z, name='z_lrelu')
	print ("Z shape:")
	print (z.get_shape())

	# Decoder network
	n_output = 640 * 480
	limit = math.sqrt(6.0 / (n_output))
	W = tf.Variable(tf.random_uniform([numberOfTemplates, n_output], -limit, limit))
	y = tf.matmul(z, W)
	y = tf.reshape(y, [-1, 480, 640, 1], name="output")

	# tf.summary.image('Original Image', tf.reshape(x, [-1, 480, 640, 1]), max_outputs=3)
	# tf.summary.image('Reconstructed Image', tf.reshape(y, [-1, 480, 640, 1]), max_outputs=3)
	tf.summary.image('Original Image', x, max_outputs=3)
	tf.summary.image('Reconstructed Image', y, max_outputs=3)

	return {'x': x, 'z': z, 'y': y}

# Dimensions should start from 640x480 (307,200), 320x240 (76,800), 160x120 (19,200), 80x60 (4,800), 40x30 (1,200), 20x15 (300)
def dense_autoencoder(x, phase, dimensions=[307200, 512, 64]):
	current_input = tf.contrib.layers.flatten(x)

	# %% Build the encoder
	encoder = []
	for layer_i, n_output in enumerate(dimensions[1:]):
		with tf.variable_scope("encoder_" + str(layer_i + 1)):
			n_input = int(current_input.get_shape()[1])
			limit = math.sqrt(6.0 / (n_input + n_output))
			W = tf.Variable(tf.random_uniform([n_input, n_output], -limit, limit))
			b = tf.Variable(tf.zeros([n_output]))
			encoder.append(W)
			out = tf.layers.batch_normalization(inputs=tf.matmul(current_input, W) + b, center=True, scale=True, training=phase, name='encoder_' + str(layer_i + 1) + '_bn')
			output = lrelu(out, name="encoder_lrelu_" + str(layer_i + 1))
			# output = lrelu(tf.matmul(current_input, W) + b, name="encoder_lrelu_" + str(layer_i + 1))
			current_input = output

	# %% latent representation
	z = current_input
	encoder.reverse()

	# %% Build the decoder using the SAME weights
	lastIndex = len(encoder) - 1
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
		# with tf.variable_scope("encoder_" + str(layer_i + 1)):
		W = tf.transpose(encoder[layer_i])
		b = tf.Variable(tf.zeros([n_output]))
		# Add non-linearity if not the last layer
		if layer_i != lastIndex:
			out = tf.layers.batch_normalization(inputs=tf.matmul(current_input, W) + b, center=True, scale=True, training=phase, name='decoder_' + str(layer_i + 1) + '_bn')
			output = lrelu(out, name="decoder_lrelu_" + str(layer_i + 1))
			# output = lrelu(tf.matmul(current_input, W) + b, name="decoder_lrelu_" + str(layer_i + 1))
		else:
			output = tf.matmul(current_input, W) + b
		current_input = output

	# %% now have the reconstruction through the network
	y = tf.reshape(current_input, [-1, 480, 640, 1], name="output")

	tf.summary.image('Original Image', x, max_outputs=3)
	tf.summary.image('Reconstructed Image', y, max_outputs=3)

	return {'x': x, 'z': z, 'y': y}

# %%
def autoencoder(x,
				n_filters=[1, 10, 10, 10, 10, 10],
				filter_sizes=[3, 3, 3, 3, 3, 3]):
	"""Build a deep autoencoder w/ tied weights.
	Parameters
	----------
	input_shape : list, optional
		Description
	n_filters : list, optional
		Description
	filter_sizes : list, optional
		Description
	Returns
	-------
	x : Tensor
		Input placeholder to the network
	z : Tensor
		Inner-most latent representation
	y : Tensor
		Output reconstruction of the input
	cost : Tensor
		Overall cost to use for training
	Raises
	------
	ValueError
		Description
	"""
	current_input = x

	# Build the encoder
	strides = [2, 2, 2, 2, 2]
	encoder = []
	shapes = []
	weight_decay_factor = 5e-4
	for layer_i, n_output in enumerate(n_filters[1:]):
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())

		# stddev = (2 / n_input)**0.5
		# shape = [filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output]
		# initializer = tf.truncated_normal_initializer(stddev=stddev)
		# W = tf.get_variable('W_' + str(layer_i), shape=shape, initializer=initializer)
		# weight_decay = tf.multiply(tf.nn.l2_loss(W), weight_decay_factor, name='weight_loss_' + str(layer_i))
		# tf.add_to_collection('losses', weight_decay)
		W = tf.Variable(
			tf.random_uniform([
			# tf.truncated_normal([
				filter_sizes[layer_i],
				filter_sizes[layer_i],
				n_input, n_output],
				-1.0 / math.sqrt(n_input),
				1.0 / math.sqrt(n_input)))
				# mean = 0,
				# stddev = math.sqrt(2.0 / n_input)))
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)
		output = lrelu(
			tf.add(tf.nn.conv2d(
				current_input, W, strides=[1, strides[layer_i], strides[layer_i], 1], padding='SAME'), b), 
				name="lrelu_" + str(layer_i))

		current_input = output

	# %%
	# store the latent representation
	z = current_input
	encoder.reverse()
	shapes.reverse()
	strides.reverse()

	# %%
	# Build the decoder using the SAME weights
	lastIndex = len(shapes) - 1
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		output = tf.add(
					tf.nn.conv2d_transpose(
					current_input, W,
					tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
					strides=[1, strides[layer_i], strides[layer_i], 1], padding='SAME'), b)
		if layer_i == lastIndex:
			# output = tf.sigmoid(output)
			output = tf.identity(lrelu(output, name="lrelu_" + str(lastIndex)), name="output") # No activation function

		else:
			output = lrelu(output, name="lrelu_" + str(layer_i + len(filter_sizes) - 1))
		# output = lrelu(tf.add(
		#     tf.nn.conv2d_transpose(
		#         current_input, W,
		#         tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
		#         strides=[1, 2, 2, 1], padding='SAME'), b))
		current_input = output

	# %%
	# now have the reconstruction through the network
	y = current_input
	# y.set_shape((None, 480, 640, 1))

	tf.summary.image('Original Image', x, max_outputs=3)
	tf.summary.image('Reconstructed Image', y, max_outputs=3)

	# %%
	return {'x': x, 'z': z, 'y': y}


# %%
def autoencoder_complete(x,
				n_filters=[1, 3, 10, 10, 3],
				filter_sizes=[3, 3, 3, 3],
				dimensions=[1200*3, 256]):
	"""Build a deep autoencoder w/ tied weights.
	Parameters
	----------
	input_shape : list, optional
		Description
	n_filters : list, optional
		Description
	filter_sizes : list, optional
		Description
	Returns
	-------
	x : Tensor
		Input placeholder to the network
	z : Tensor
		Inner-most latent representation
	y : Tensor
		Output reconstruction of the input
	cost : Tensor
		Overall cost to use for training
	Raises
	------
	ValueError
		Description
	"""
	current_input = x

	# Build the encoder
	strides = [2, 2, 2, 2]
	encoder = []
	shapes = []
	weight_decay_factor = 5e-4
	for layer_i, n_output in enumerate(n_filters[1:]):
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())
		W = tf.Variable(
			tf.random_uniform([
				filter_sizes[layer_i],
				filter_sizes[layer_i],
				n_input, n_output],
				-1.0 / math.sqrt(n_input),
				1.0 / math.sqrt(n_input)))
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)
		output = lrelu(
			tf.add(tf.nn.conv2d(
				current_input, W, strides=[1, strides[layer_i], strides[layer_i], 1], padding='SAME'), b), 
				name="lrelu_" + str(layer_i))

		current_input = output

	############ Shallow autoencoder - Start ############
	# %% Build the encoder
	
	# Flatten the array
	dataShape = current_input.get_shape().as_list()
	print(dataShape)
	current_input = tf.reshape(current_input, [-1, dataShape[1] * dataShape[2] * dataShape[3]])

	encoder_shallow = []
	for layer_i, n_output in enumerate(dimensions[1:]):
		n_input = int(current_input.get_shape()[1])
		W = tf.Variable(
			tf.random_uniform([n_input, n_output],
							  -1.0 / math.sqrt(n_input),
							  1.0 / math.sqrt(n_input)))
		b = tf.Variable(tf.zeros([n_output]))
		encoder_shallow.append(W)
		output = tf.nn.tanh(tf.matmul(current_input, W) + b)
		current_input = output

	# %% latent representation
	z_shallow = current_input
	encoder_shallow.reverse()

	# %% Build the decoder using the SAME weights
	lastIndex = len(encoder_shallow) - 1
	for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
		W = tf.transpose(encoder_shallow[layer_i])
		b = tf.Variable(tf.zeros([n_output]))
		output = tf.nn.tanh(tf.matmul(current_input, W) + b)
		current_input = output

	# Restore array shape
	current_input = tf.reshape(current_input, [-1, dataShape[1], dataShape[2], dataShape[3]])

	############ Shallow autoencoder - End ############

	# %%
	# store the latent representation
	# z = current_input
	encoder.reverse()
	shapes.reverse()
	strides.reverse()

	# %%
	# Build the decoder using the SAME weights
	lastIndex = len(shapes) - 1
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		output = tf.add(
					tf.nn.conv2d_transpose(
					current_input, W,
					tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
					strides=[1, strides[layer_i], strides[layer_i], 1], padding='SAME'), b)
		if layer_i == lastIndex:
			# output = tf.sigmoid(output)
			output = tf.identity(lrelu(output, name="lrelu_" + str(lastIndex)), name="output") # No activation function

		else:
			output = lrelu(output, name="lrelu_" + str(layer_i + len(filter_sizes) - 1))
		# output = lrelu(tf.add(
		#     tf.nn.conv2d_transpose(
		#         current_input, W,
		#         tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
		#         strides=[1, 2, 2, 1], padding='SAME'), b))
		current_input = output

	# %%
	# now have the reconstruction through the network
	y = current_input
	# y.set_shape((None, 480, 640, 1))

	tf.summary.image('Original Image', x, max_outputs=3)
	tf.summary.image('Reconstructed Image', y, max_outputs=3)

	# %%
	# return {'x': x, 'z': z, 'y': y}
	return {'x': x, 'z': z_shallow, 'y': y}
