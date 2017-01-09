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

# dropout: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
# how to get rid of dropout after training

# decaying learning rate: https://www.tensorflow.org/api_docs/python/train/decaying_the_learning_rate


def five_layer_network(x, dropout):
    print "training 5 layer deep neural network"

    a, b, c, d = 400, 200, 100, 50

    w1 = tf.Variable(tf.truncated_normal([784, a], stddev=0.1),  name="w1")
    b1 = tf.Variable(tf.zeros([a]), name="b1")
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1, name="y1")

    w2 = tf.Variable(tf.truncated_normal([a, b], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.zeros([b]), name="b2")
    y2 = tf.nn.relu(tf.matmul(y1, w2) + b2, name="y2")

    w3 = tf.Variable(tf.truncated_normal([b, c], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.zeros([c]), name="b3")
    y3 = tf.nn.relu(tf.matmul(y2, w3) + b3, name="y3")

    w4 = tf.Variable(tf.truncated_normal([c, d], stddev=0.1), name="w4")
    b4 = tf.Variable(tf.zeros([d]), name="b4")
    y4 = tf.nn.relu(tf.matmul(y3, w4) + b4, name="y4")

    w5 = tf.Variable(tf.truncated_normal([d, 10], stddev=0.1), name="w5")
    b5 = tf.Variable(tf.zeros([10]), name="b5")
    y = tf.matmul(y4, w5) + b5

    # applying dropout
    y_drop = tf.nn.dropout(y, dropout)

    return y_drop


def train():

    dropout_prob = .75 # probability of keeping neuron when training (when testing for accuracy, p_keep should be 1)
    # dropout probability of model; we can feed in dropout_prob when training and 1 for testing
    p_keep = tf.placeholder(tf.float32, name="p_keep")

    x = tf.placeholder(tf.float32, [None, 784], name="x")

    y = five_layer_network(x, p_keep)

    correct_labels = tf.placeholder(tf.float32, [None, 10], name="correct_labels")

    # cost function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, correct_labels))

    # applying learning rate decay MAYBE TRY ADAM OPTIMIZER?
    decay_steps = 1000
    decay_rate = 0.96
    global_step = tf.Variable(0, trainable=False) # trainable=False means the optimizer can't change the variable
    starting_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(
        starting_learning_rate, global_step=global_step, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    num_epochs = 14
    batch_size = 100

    with tf.Session() as sess:
        tf.set_random_seed(1)
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        sess.run(init)

        # for evaluating accuracy on test data
        correct = tf.equal(tf.argmax(y, 1), tf.argmax(correct_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([training_step, cross_entropy], feed_dict= {x: batch_x, correct_labels: batch_y,
                                                                            # we're applying dropout on neurons
                                                                            p_keep: dropout_prob
                                                                            })
                epoch_loss += c
            print('EPOCH: ' + str(epoch) + ' out of ' + str(num_epochs))
            print('\tloss: ' + str(epoch_loss))
            # accuracy calculated from test data
            print('\taccuracy: ' + str(accuracy.eval({x:mnist.test.images, correct_labels: mnist.test.labels,
                                                      # dropout probability is 1 so no neurons are dropped for testing
                                                      p_keep: 1})))

        print('Final Accuracy: ' + str(sess.run([accuracy],
                                feed_dict={x: mnist.test.images, correct_labels: mnist.test.labels, p_keep: 1})[0]))

train()

# questions: truncated normal and stddev vals, optimal neural net configuration (values for a, b, c and how do they work)

# how does this work?
# # Optimizer: set up a variable that's incremented once per batch and
# # controls the learning rate decay.
# batch = tf.Variable(0)
#
# learning_rate = tf.train.exponential_decay(
#   0.01,                # Base learning rate.
#   batch * BATCH_SIZE,  # Current index into the dataset.
#   train_size,          # Decay step.
#   0.95,                # Decay rate.
#   staircase=True)
# # Use simple momentum for the optimization.
# optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                      0.9).minimize(loss,
#                                                    global_step=batch)
