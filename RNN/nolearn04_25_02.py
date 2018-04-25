"""

自编码网络
1、自编码网络的作用
自编码网络的作用就是将输入样本压缩到隐藏层，然后解压，在输出端重建样本，最终输出层神经元数量等于输入层神经元的数量。
2、这里主要有两个过程，压缩和解压。
3、压缩原理
压缩依靠的是输入数据（图像、文字、声音）本身存在不同成都的冗余信息，自动编码网络学习去掉这些冗余信息，把有用的特征输入到隐藏层中。
4、多个隐藏层的主要作用
多个隐藏层的主要作用是，如果输入的数据是图像，第一层会学习如何识别边，第二层会学习如何组合边，从而构成轮廓、角等，更高层学习如何去组合更有意义的特征。
5、下面我们还以MINST数据集为例，讲解一下自编码器的运用

"""


from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

# Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
# hidden layer settings
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features

num_input = 784
weights = {
'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2
# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Launch the graph
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("第{0}次训练".format(epoch),"Epoch:", '%04d' % (epoch+1),"loss值=", "{:.9f}".format(c))
    print("Optimization Finished!")
    # # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()
    # encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()

