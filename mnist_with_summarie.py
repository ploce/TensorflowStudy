#coding:utf-8
#导入测试数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#导入tensorflow
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])

with tf.name_scope('input'):
    image_shaped_input = tf.reshape(x, [-1, 28,28,1])
    tf.image_summary('input', image_shaped_input, 50)

#设置变量，这个是需要学习的参数
with tf.name_scope('parameter'):
    with tf.name_scope('W'):
        W = tf.Variable(tf.zeros([784,10]))
    with tf.name_scope('b'):
        b = tf.Variable(tf.zeros([10]))

#模型输出以及正确结果
with tf.name_scope('output'):
    with tf.name_scope('y'):
        y = tf.nn.softmax(tf.matmul(x,W) + b)
    with tf.name_scope('y_'):
        y_ = tf.placeholder("float", [None,10])

#交叉熵，训练的目标就是最小化该值
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    tf.scalar_summary('correct_prediction', cross_entropy)

#训练模型，使用梯度下降法，以最小化交叉熵为目标
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#sess = tf.Session()
sess = tf.InteractiveSession()

#设置结果处理的先相关内容
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary('accuracy', accuracy)

#创建会话（Session），并初始化参数
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('/tmp/mnist_logs/train', sess.graph)
init = tf.initialize_all_variables()
#sess = tf.Session()
sess.run(init)

#循环训练1000次，每次随机选取100组数据
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summary,acc,_ = sess.run([merged, accuracy, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    train_writer.add_summary(summary, i)
    print('Accuracy at step %s : %s' % (i, acc))

#运行测试数据，并打印结果
#print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
