#coding=utf-8
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 1000,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)
    
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,cluster=cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            x = tf.placeholder(tf.float32, [None, 784])
            y_ = tf.placeholder("float", [None,10])

            #卷积与池化
            W_conv1 = weight_variable([5,5,1,32])
            b_conv1 = bias_variable([32])

            x_image = tf.reshape(x, [-1,28,28,1])

            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            W_conv2 = weight_variable([5,5,32,64])
            b_conv2 = bias_variable([64])

            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            W_fc1 = weight_variable([7*7*64, 1024])
            b_fc1 = bias_variable([1024])

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            keep_prob = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2 = weight_variable([1024,10])
            b_fc2 = bias_variable([10])


            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step = global_step)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            init_op = tf.global_variables_initializer()
            
            saver = tf.train.Saver()

            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                    logdir="./checkpoint/",
                    init_op=init_op,
                    summary_op=None,
                    saver=saver,
                    global_step=global_step,
                    save_model_secs=60)
        
            with sv.managed_session(server.target) as sess:
                step = 0
                while  step < 20000:
                    batch_xs, batch_ys = mnist.train.next_batch(100)
                    _, acc, step = sess.run([train_step, accuracy, global_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})

                    print("step: %d, Accuracy: %s" %(step, acc))
            #if FLAGS.task_index == 0:
            #    print "test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.image, y_:mnist.test.labels, keep_prob: 1.0})
            
            sv.stop()

if __name__ == "__main__":
  tf.app.run()
