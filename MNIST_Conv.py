# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:51:06 2019

@author: crzzzhang
"""

import tensorflow as tf
import input_data

mnist=input_data.read_data_sets('D:\Dataset\MNIST',one_hot=True)


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1) #平均值和标准差可以设定的正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding='VALID')

x=tf.placeholder('float',[None,784])    
y_=tf.placeholder('float',[None,10])    #标签y的真实值

W_conv1=weight_variable([5,5,1,32]) 
b_conv1=bias_variable([32])

x_image=tf.reshape(x,[-1,28,28,1])  
#shape里最多只能有一个-1，-1处的实际值保证reshape前后shape的乘积不变
#x为图像数据，reshape后的shape为None,28,28,1
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
#None,14,14,32

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)
#None,7,7,64

W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#None,1024

keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#设置dropout，设置为占位符，则可以在训练过程中启用dropout，在准确性测试时关掉

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#None,10

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#损失函数，交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#用Adam优化器来做梯度下降
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#None,1（BOOL）
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#通过看onehot标签的预测值与实际值得到准确率，cast是把bool转换为float

with tf.Session() as sess:  #当用到eval来看值的时候，需要传递sess或者像这样用with
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(64)
      if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

