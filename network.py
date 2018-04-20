# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:08
# @Author  : jiaopan
# @Email   : jiaopaner@163.com
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
import  utils
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', int(utils.configUtil("global.conf","dataset","batch_size")),"""每个batch样本总数""")
IMAGE_SIZE = int(utils.configUtil("global.conf","dataset","image_mat_size"))
NUM_CLASSES = int(utils.configUtil("global.conf","dataset","num_class"))
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = int(utils.configUtil("global.conf","train","train_data_count"))
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(utils.configUtil("global.conf","eval","eval_data_count"))

# 超参数设置
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1

TOWER_NAME = 'tower'

def activation_summary(x):
  """创建tensorboard摘要."""
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def variable_on_cpu(name, shape, initializer):
  """针对CPU-train 初始化参数."""
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def variable_with_weight_decay(name, shape, stddev, wd):
  """初始化 Variable with weight decay"""
  var = variable_on_cpu(name, shape,tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images):
  """构建模型.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  """
  # tf.get_variable() 可参数共享 针对multiple GPU
  # tf.Variable()  single GPU
  #
  # 卷积层1
  with tf.variable_scope('conv1') as scope:
    #初始化卷积核
    kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                         stddev=1e-4, wd=0.0)
    #卷积操作
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    activation_summary(conv1)

  # 池化层1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # 局部响应标准化
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')
  # 卷积层2
  with tf.variable_scope('conv2') as scope:
    kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                         stddev=1e-4, wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    activation_summary(conv2)

  # 局部响应标准化
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # 池化层2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  ###############全连接层################
  #local3
  with tf.variable_scope('local3') as scope:
    dim = 1
    for d in pool2.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
    weights = variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    activation_summary(local3)
  # local4
  with tf.variable_scope('local4') as scope:
    weights = variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    activation_summary(local4)
  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=0.0)
    biases = variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    activation_summary(softmax_linear)

  return softmax_linear

def loss(logits, labels):
  """L2正则的误差计算."""
  sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
  indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
  concated = tf.concat([indices, sparse_labels], 1)
  dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, NUM_CLASSES],1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = dense_labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def add_loss_summaries(total_loss):
  """创建误差tesorboard摘要"""
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))
  return loss_averages_op

def train(total_loss, global_step):
  """训练模型."""

  #每个epoch的batch数
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
  # 学习速率的配置
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  loss_averages_op = add_loss_summaries(total_loss)

  # 梯度下降
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  return train_op