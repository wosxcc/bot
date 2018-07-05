import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.python.framework import graph_util


w = 96
h = 96
c = 3

# def read_img(path):
#     cate   = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
#     imgs   = []
#     labels = []
#     for idx, folder in enumerate(cate):
#         for im in glob.glob(folder + '/*.jpg'):
#             print('reading the image: %s' % (im))
#             img = io.imread(im)
#             img = transform.resize(img, (w, h, c))
#             imgs.append(img)
#             labels.append(idx)
#     return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

txt_name= 'trains.txt'
def read_img(txt_name):
    label_lines = []
    image_lines = []
    txt_open = open(txt_name)
    txt_read = txt_open.read()
    txt_lines = txt_read.split('\n')

    for line in txt_lines:
        xlabel = []
        if len(line)>3:
            line_list = line.split(' ')
            image_lines.append(cv.imread(line_list[0]))
            xlabel.append(line_list[1])
            xlabel.append(line_list[2])
            for x in range(14):
                xlabel.append(line_list[117 + 2 + x * 2])
                xlabel.append(line_list[117 + 2 + x * 2 + 1])
            label_lines.append(xlabel)

    label_linesc=[[float(i) for i in xline] for xline in label_lines]
    ximage_lines=np.array(image_lines, dtype='float32')
    ximage_lines/=255
    xlabel_linesc=np.array(label_linesc, dtype='float32')
    return ximage_lines,xlabel_linesc





data, label = read_img('trains.txt')
print(data.shape)
num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

ratio = 0.9
s = np.int(num_example * ratio)
x_train = data[:s]
y_train = label[:s]
x_val   = data[s:]
y_val   = label[s:]

def build_network(height, width, channel):
    x = tf.placeholder(tf.float32, shape=[None, height, width, channel], name='input')
    y = tf.placeholder(tf.float32, shape=[None, 30], name='labels_placeholder')

    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(input, w, c):
        return tf.nn.conv2d(input, w, c, padding='SAME')

    def pool_max(input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def fc(input, w, b):
        return tf.matmul(input, w) + b
    c1=[1,1,1,1]
    c2=[1,2,2,1]
    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 3, 32])
        biases = bias_variable([32])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel,c1) + biases, name=scope)

    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 32, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel,c2) + biases, name=scope)

    # pool1 = pool_max(output_conv1_2)

    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(output_conv1_2, kernel,c1) + biases, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel,c2) + biases, name=scope)

    # pool2 = pool_max(output_conv2_2)

    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 256, 128])
        biases = bias_variable([128])
        output_conv3_1 = tf.nn.relu(conv2d(output_conv2_2, kernel,c1) + biases, name=scope)

    # with tf.name_scope('conv3_2') as scope:
    #     kernel = weight_variable([3, 3, 256, 256])
    #     biases = bias_variable([256])
    #     output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)
    #
    # with tf.name_scope('conv3_3') as scope:
    #     kernel = weight_variable([3, 3, 256, 256])
    #     biases = bias_variable([256])
    #     output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)
    #
    # pool3 = pool_max(output_conv3_3)
    #
    # # conv4
    # with tf.name_scope('conv4_1') as scope:
    #     kernel = weight_variable([3, 3, 256, 512])
    #     biases = bias_variable([512])
    #     output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)
    #
    # with tf.name_scope('conv4_2') as scope:
    #     kernel = weight_variable([3, 3, 512, 512])
    #     biases = bias_variable([512])
    #     output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)
    #
    # with tf.name_scope('conv4_3') as scope:
    #     kernel = weight_variable([3, 3, 512, 512])
    #     biases = bias_variable([512])
    #     output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)
    #
    # pool4 = pool_max(output_conv4_3)
    #
    # with tf.name_scope('conv5_1') as scope:
    #     kernel = weight_variable([3, 3, 512, 512])
    #     biases = bias_variable([512])
    #     output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)
    #
    # with tf.name_scope('conv5_2') as scope:
    #     kernel = weight_variable([3, 3, 512, 512])
    #     biases = bias_variable([512])
    #     output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)
    #
    # with tf.name_scope('conv5_3') as scope:
    #     kernel = weight_variable([3, 3, 512, 512])
    #     biases = bias_variable([512])
    #     output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)
    #
    # pool5 = pool_max(output_conv5_3)

    #fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(output_conv3_1.get_shape()[1:]))
        kernel = weight_variable([shape, 256])
        biases = bias_variable([256])
        pool5_flat = tf.reshape(output_conv3_1, [-1, shape])
        output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)

    # #fc7
    # with tf.name_scope('fc7') as scope:
    #     kernel = weight_variable([4096, 4096])
    #     biases = bias_variable([4096])
    #     output_fc7 = tf.nn.relu(fc(output_fc6, kernel, biases), name=scope)

    #fc8
    with tf.name_scope('fc8') as scope:
        kernel = weight_variable([256, 30])
        biases = bias_variable([30])
        output_fc8 = tf.nn.relu(fc(output_fc6, kernel, biases), name='output')

    # finaloutput = tf.nn.softmax(output_fc8, name="softmax")

    cost  = tf.sqrt(tf.reduce_mean(tf.square(output_fc8 - y)))

    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost, global_step=global_step)
    # optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    # prediction_labels = tf.argmax(finaloutput, axis=1, name="output")
    # read_labels = y
    #
    # correct_prediction = tf.equal(prediction_labels, read_labels)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

    return dict(
        x=x,
        y=y,
        optimize=optimize,
        # correct_prediction=correct_prediction,
        # correct_times_in_batch=correct_times_in_batch,
        cost=cost,
    )


def train_network(graph, batch_size, num_epochs, pb_file_path):
    print(graph)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        epoch_delta = 50
        for epoch_index in range(num_epochs):
            for i in range(9555):

                sess.run([graph['optimize']], feed_dict={
                    graph['x']: np.reshape(x_train[i], (1, 96, 96, 3)),
                    graph['y']: np.reshape(y_train[i], (1, 30))

                })
            if epoch_index % epoch_delta == 0:
                total_batches_in_train_set = 0
                # total_correct_times_in_train_set = 0
                total_cost_in_train_set = 0.
                for i in range(12):
                    # return_correct_times_in_batch = sess.run(graph['cost'], feed_dict={
                    #     graph['x']: np.reshape(x_train[i], (1, 224, 224, 3)),
                    #     graph['y']: ([[1, 0]] if y_train[i] == 0 else [[0, 1]])
                    # })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_train[i], (1, 96, 96, 3)),
                        graph['y']: np.reshape(y_train[i], (1, 30))
                    })
                    total_batches_in_train_set += 1
                    # total_correct_times_in_train_set += return_correct_times_in_batch
                    total_cost_in_train_set += (mean_cost_in_batch * batch_size)


                total_batches_in_test_set = 0
                total_cost_in_test_set = 0.
                for i in range(3):
                    # return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                    #     graph['x']: np.reshape(x_val[i], (1, 224, 224, 3)),
                    #     graph['y']: ([[1, 0]] if y_val[i] == 0 else [[0, 1]])
                    # })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x']: np.reshape(x_val[i], (1, 96, 96, 3)),
                        graph['y']: np.reshape(y_train[i], (1, 30))
                    })
                    total_batches_in_test_set += 1
                    # total_correct_times_in_test_set += return_correct_times_in_batch
                    total_cost_in_test_set += (mean_cost_in_batch * batch_size)

                # acy_on_test  = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                # acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                print('第几次：',epoch_index, '总loss',total_batches_in_test_set * batch_size,'测试loss',total_cost_in_test_set,
                                           '训练loss',total_cost_in_train_set)
            if epoch_index%100==1:
                print('第几次：', epoch_index,'进行保存')
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["fc8/output"])
                with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())


def main():
    batch_size = 12
    num_epochs = 50000

    pb_file_path = "vggs.pb"

    g = build_network(height=96, width=96, channel=3)
    train_network(g, batch_size, num_epochs, pb_file_path)

main()
