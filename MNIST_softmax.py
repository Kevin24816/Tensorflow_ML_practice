import tensorflow as tf
import pickle
from tensorflow.examples.tutorials.mnist import input_data

# saving/loading mnist data
try:
    mnist = pickle.load(open("MNIST_save.p", "rb"))
except:
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    print "pickling"
    pickle.dump(mnist, open("MNIST_save.p", "wb"))

# initializing tensorflow variables
def train():
    ###############DEFINE_TRAINING_PARAMETERS############
    # holds an array of images, each defined as a 784x1 array
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    # matrix storing weights for 784 points (the 28x28 image flattened out) by 10 classes (one for each digit)
    W = tf.Variable(tf.zeros([784, 10]), name="weight")
    # 10x1 matrix contain the bias for each class (that will be added to the final weight)
    b = tf.Variable(tf.zeros([10]), name="bias")
    # the resulting y matrix value prior to the activation function
    y = tf.matmul(x, W) + b

    # holds the final y-matrix value after softmax activation
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    ###############DEFINING_OPTIMIZATION###########################
    # training step using gradient descent optimizer with learning rate 0.5 minimizing loss calculated by cross entropy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    tstep = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    ###############TRAINING_DATA########################
    # initializing session and variables
    s = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    num_it = 1000 # running 1000 epochs
    bs = 100 # batch size, or number of pictures we are feeding the neural net at one time

    for epoch in range(num_it):
        # getting random batches (size bs) of x (images) and y (labels)
        batch_x, batch_y = mnist.train.next_batch(bs)
        # feeding the batches into the session
        s.run(tstep, feed_dict= {x: batch_x, y_: batch_y})

     # compares the predicted value to actual value and returns array of booleans on whether the prediction was true
    compare_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # calculates the accuracy on the number of test cases in which the prediction is true
    accuracy = tf.reduce_mean(tf.cast(compare_prediction, tf.float32))

    # now runs the trained model on test data, calculating the accuracy along the way
    print(s.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

train()
