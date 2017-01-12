import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data

# saving/loading mnist data
def getMNIST():
    try:
        return pickle.load(open("data/MNIST.p", "rb"))
    except:
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        print "pickling"
        pickle.dump(mnist, open("data/MNIST.p", "wb"))
        return mnist

mnist = getMNIST()


def build_conv_layers(x):
    # Since we're learning on 2D image, let's reshape x from 1x784 to 28x28
    x = tf.reshape(x, [-1, 28, 28, 1], name="reshaped_x")

    # num output channels
    a, b, c = 6, 12, 24

    # filter sizes
    f1, f2 = 5, 4

    # 4d array for convolutional layer with f1xf1 filter, 1 input channel (black and white), and 'a' output channels
    w1 = tf.Variable(tf.truncated_normal([f1, f1, 1, a], stddev=0.1))
    b1 = tf.Variable(tf.ones([a])/10)  # 1D array storing biases for each channel

    w2 = tf.Variable(tf.truncated_normal([f1, f1, a, b], stddev=0.1))
    b2 = tf.Variable(tf.ones([b])/10)

    w3 = tf.Variable(tf.truncated_normal([f2, f2, b, c], stddev=0.1))
    b3 = tf.Variable(tf.ones([c])/10)

    # weights for fully connected layer with output of length n
    n = 200
    w4 = tf.Variable(tf.truncated_normal([7*7*c, n], stddev=0.1))
    b4 = tf.Variable(tf.ones([n])/100)

    # softmax readout layer
    w5 = tf.Variable(tf.truncated_normal([n, 10], stddev=0.1))
    b5 = tf.Variable(tf.zeros([10])/100)

    # calculating outputs for convolutional layers
    #   stride of [1, 1, 1, 1]: 1 image with filter 1x1 and 1 channel
    #   padding of 'SAME' retains the same filter shape when exceeding image border
    y1 = tf.nn.relu(tf.add(
            tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME'), b1), name='y1') # stride of 1
    y2 = tf.nn.relu(tf.add(
            tf.nn.conv2d(y1, w2, strides=[1, 2, 2, 1], padding='SAME'), b2), name='y2') # stride of 2
    y3 = tf.nn.relu(tf.add(
            tf.nn.conv2d(y2, w3, strides=[1, 2, 2,1], padding='SAME'), b3), name='y3') # stride of 2

    flat_y = tf.reshape(y3, shape=[-1, 7 * 7 * c], name='flat_y')  # flatten y3 for the fully connected layer

    y4 = tf.nn.relu(tf.add(tf.matmul(flat_y, w4), b4), name='y4')  # fully connected layer
    final_y = tf.add(tf.matmul(y4, w5), b5, name='final_y')

    return final_y

def train():
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = build_conv_layers(x)

    # adding dropout
    p_keep = tf.placeholder(tf.float32, name="dropout")  # used to substitute dropout prob during training/testing
    dropout_prob = .75
    y_drop = tf.nn.dropout(y, p_keep)

    correct_labels = tf.placeholder(tf.float32, [None, 10], name="correct_labels")

    # defining loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_drop, correct_labels))

    # training step
    learning_rate = 0.001 # will decrease with adam optimizer
    training_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # for evaluating accuracy on test data
    correct = tf.equal(tf.argmax(y_drop, 1), tf.argmax(correct_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    init = tf.global_variables_initializer()

    # training session
    with tf.Session() as sess:
        tf.set_random_seed(1)
        sess.run(init)

        step = 1
        training_iters = 1000000
        batch_size = 130
        display_step = 10

        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) on training data
            sess.run(training_step, feed_dict={x: batch_x, correct_labels: batch_y, p_keep: dropout_prob})
            if step % display_step == 0:
                # Calculate batch loss and accuracy on training data
                loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, correct_labels: batch_y,
                                                                           p_keep: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) +
                      ", Training Accuracy: " + "{:.5f}".format(acc))

                # Calculate accuracy on testing data
                print('\tTesting Accuracy: ' + str(accuracy.eval({x:mnist.test.images, correct_labels: mnist.test.labels,
                                                      p_keep: 1})))
            step += 1

        print("Training complete!")

train()

# notes: how to use tf.reshape()
