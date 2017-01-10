#coding:utf-8
#导入测试数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#导入tensorflow
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])

#设置变量，这个是需要学习的参数
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#模型输出以及正确结果
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])

#交叉熵，训练的目标就是最小化该值
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#训练模型，使用梯度下降法，以最小化交叉熵为目标
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#创建会话（Session），并初始化参数
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#循环训练1000次，每次随机选取100组数据
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#设置结果处理的先相关内容
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#运行测试数据，并打印结果
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
