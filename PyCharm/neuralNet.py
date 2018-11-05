import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# hyperparameter festlegen
# Number of neurons per Layer
n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

# Unterschiedliche Klassen, Zahlen 0 - 9
n_classes = 10

# Wie viele Bilder auf einmal gelesen werden können
batch_size = 100

# 28 x 28 pixel pro Bild
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, n_classes])


def neuralNetworkModel(data):

    # per Zufallszahlen trainieren
    hiddenLayer_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_h1])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_h1]))}

    hiddenLayer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_h2]))}

    hiddenLayer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])),
                     'biases': tf.Variable(tf.random_normal([n_nodes_h3]))}

    outputLayer = {'weights': tf.Variable(tf.random_normal([n_nodes_h3, n_classes])),
                     'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Matrix multiplication
    l1 = tf.add(tf.matmul(data, hiddenLayer_1['weights']), hiddenLayer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hiddenLayer_3['weights']), hiddenLayer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, outputLayer['weights']) + outputLayer['biases']

    return output



def trainNeuralNetwork(x):
    prediction = neuralNetworkModel(x)
    # Cost minimieren (Fehlerrate)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    #
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Epochs
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for ep in range(epochs):
            epochs_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epochs_x, epochs_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimizer], feed_dict={x: epochs_x, y: epochs_y})
                epochs_loss+=c

                print('Epochs',  ep, 'compleded out of ', epochs, 'loss', epochs_loss)

            correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


trainNeuralNetwork(x)















