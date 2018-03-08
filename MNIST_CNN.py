import numpy as np
import tensorflow as tf
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs_CNN"
logdir = "{}/run-{}".format(root_logdir, now)

tf.reset_default_graph()

N_INPUTS = 28*28
DROPOUT_RATE = 0.4
n_outputs = 10

LEARNING_RATE = 0.01
MOMENTUM = 0.9
USE_NESTEROV = True

N_EPOCHS = 1000
BATCH_SIZE = 500
MAX_CHECKS_WITHOUT_PROGRESS = 10

X = tf.placeholder(tf.float32, shape=[None, N_INPUTS], name="X")
y = tf.placeholder(tf.int64, shape=[None], name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

# cl1 = Convolutional Layer 1
# pl1 = pooling Layer 1
# dl1 = dense Layer 1

input_layer = tf.reshape(X, [-1, 28, 28, 1])
cl1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=5,
    strides=[1,1],
    padding="same",
    activation=tf.nn.relu,
    name="convolution_1")
pl1 = tf.layers.max_pooling2d(cl1, pool_size=[2,2], strides=[2,2], padding="valid", name="pooling_1")
cl2 = tf.layers.conv2d(
    inputs=pl1,
    filters=64,
    kernel_size=5,
    strides=[1,1],
    padding="same",
    activation=tf.nn.relu,
    name="convolution_2")
pl2 = tf.layers.max_pooling2d(cl2, pool_size=[2,2], strides=[2,2], padding="valid", name="pooling_2")
# print(pl2)
pl2_flat = tf.reshape(pl2, [-1, 7 * 7 * 64])
dl1 = tf.layers.dense(pl2_flat, 1024, activation=tf.nn.relu, name="dense_1")
dl1_drop = tf.layers.dropout(dl1, rate=DROPOUT_RATE, training=training, name="dropout")
dl2 = tf.layers.dense(dl1_drop, 10, name="dense_2")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=dl2, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(LEARNING_RATE, MOMENTUM, use_nesterov=USE_NESTEROV)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(dl2, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
y_train = mnist.train.labels
X_valid = mnist.validation.images
y_valid = mnist.validation.labels
X_test = mnist.test.images
y_test = mnist.test.labels

checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()

    for epoch in range(N_EPOCHS):
        rnd_idx = np.random.permutation(len(X_train))
        for rnd_indices in np.array_split(rnd_idx, len(X_train) // BATCH_SIZE):
            X_batch, y_batch = X_train[rnd_indices], y_train[rnd_indices]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        acc_train = sess.run(accuracy, feed_dict={X: X_train, y: y_train})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid, y: y_valid})
        if loss_val < best_loss:
            save_path = saver.save(sess, "./my_CNN_mnist_model.ckpt")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > MAX_CHECKS_WITHOUT_PROGRESS:
                print("Early stopping!")
                break
        print("{} \Training accuracy: {:.2f}% \tValidation loss: {:.6f}\tBest loss: {:.6f}\tValidation accuracy: {:.2f}%".format(epoch, acc_train * 100, loss_val, best_loss, acc_val * 100))
        acc_valid_summary = accuracy_summary.eval(feed_dict={X: X_valid, y: y_valid})
        summary_writer.add_summary(acc_valid_summary, epoch)

    saver.restore(sess, "./my_CNN_mnist_model.ckpt")
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final test accuracy: {:.2f}%".format(acc_test * 100))
    save_path = saver.save(sess, "./my_model_final.ckpt")