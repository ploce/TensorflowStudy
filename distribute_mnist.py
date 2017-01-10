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

            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))

            y = tf.nn.softmax(tf.matmul(x,W) + b)
            y_ = tf.placeholder("float",[None,10])

            cross_entropy = -tf.reduce_sum(y_*tf.log(y))

            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            init_op = tf.initialize_all_variables()
            
            saver = tf.train.Saver()
            tf.summary.scalar('cost', cross_entropy)
            summary_op = tf.summary.merge_all()
            
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                    logdir="./checkpoint/",
                    init_op=init_op,
                    summary_op=None,
                    saver=saver,
                    global_step=global_step,
                    save_model_secs=60)
        
        with sv.managed_session(server.target) as sess:
            step = 0
            while  step < 500:
                batch_xs, batch_ys = mnist.train.next_batch(100)
                _, acc, step = sess.run([train_step, accuracy, global_step], feed_dict={x: batch_xs, y_: batch_ys})
                
                print("step: %d, Accuracy: %s" %(step, acc))
        sv.stop()


if __name__ == "__main__":
  tf.app.run()
