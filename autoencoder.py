import tensorflow as tf
import numpy as np
import math

def lrelu(x, name, leak = 0.2):
    with tf.name_scope(name):
        return tf.maximum(x, leak * x)

# Dimensions should start from 640x480 (307,200), 320x240 (76,800), 160x120 (19,200), 80x60 (4,800), 40x30 (1,200), 20x15 (300)
def dense_autoencoder(x, dimensions=[307200, 2048, 1024, 512, 256]):
    current_input = x

    # %% Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        with tf.variable_scope("encoder_" + str(layer_i + 1)):
            n_input = int(current_input.get_shape()[1])
            # W = tf.Variable(
            #     tf.random_uniform([n_input, n_output],
            #                       -1.0 / math.sqrt(n_input),
            #                       1.0 / math.sqrt(n_input)))
            # W = tf.get_variable("W_enc_" + str(layer_i + 1), shape=[n_input, n_output],
            #                     # initializer=tf.contrib.layers.xavier_initializer())
            #                     initializer=tf.glorot_uniform_initializer())
            limit = math.sqrt(6.0 / (n_input + n_output))
            W = tf.Variable(tf.random_uniform([n_input, n_output], -limit, limit))
            b = tf.Variable(tf.zeros([n_output]))
            encoder.append(W)
            output = lrelu(tf.matmul(current_input, W) + b, name="encoder_lrelu_" + str(layer_i + 1))
            current_input = output

    # %% latent representation
    z = current_input
    encoder.reverse()

    # %% Build the decoder using the same weights
    lastIndex = len(encoder) - 1
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        # with tf.variable_scope("encoder_" + str(layer_i + 1)):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        # Add non-linearity if not the last layer
        if layer_i != lastIndex:
            output = lrelu(tf.matmul(current_input, W) + b, name="decoder_lrelu_" + str(layer_i + 1))
        else:
            output = tf.matmul(current_input, W) + b
        current_input = output

    # %% now have the reconstruction through the network
    y = tf.identity(current_input, "output")

    tf.summary.image('Original Image', tf.reshape(x, [-1, 480, 640, 1]), max_outputs=3)
    tf.summary.image('Reconstructed Image', tf.reshape(y, [-1, 480, 640, 1]), max_outputs=3)

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
    # Build the decoder using the same weights
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

    # %% Build the decoder using the same weights
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
    # Build the decoder using the same weights
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
