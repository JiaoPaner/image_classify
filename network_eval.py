#coding:utf-8
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:08
# @Author  : jiaopan
# @Email   : jiaopaner@163.com
# 模型验证模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
import network
import tfrecord
import utils
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', utils.configUtil("global.conf","eval","eval_log_dir"),
                           """验证日志目录.""")
tf.app.flags.DEFINE_string('eval_data', utils.configUtil("global.conf","eval","eval_tfrecord_dir"),
                           """验证数据集目录""")
tf.app.flags.DEFINE_string('checkpoint_dir', utils.configUtil("global.conf","model","model_dir"),
                           """保存的模型.""")
tf.app.flags.DEFINE_integer('eval_interval_secs',60*3,
                            """设置每隔多长时间做一侧评估""")
tf.app.flags.DEFINE_integer('num_examples', int(utils.configUtil("global.conf","eval","eval_data_count")),
                            """验证数据集样本总数""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """仅验证一次.""")
def eval_once(saver, summary_writer, top_k_op, summary_op):

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))#迭代次数
      true_count = 0  # 预测正确的次数
      total_sample_count = num_iter * FLAGS.batch_size #验证的样本总数
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op]) #e.g. return [true,false,true,false,false]
        true_count += np.sum(predictions)
        step += 1
      #准确率计算
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:
      coord.request_stop(e)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    float_image, label = tfrecord.eval_data_read(tfrecord_path=FLAGS.eval_data)
    images, labels = tfrecord.create_batch(float_image, label, count_num=FLAGS.num_examples)
    logits = network.inference(images)
    # tf.nn.in_top_k:计算预测的结果和实际结果的是否相等,返回bool类型的张量
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    summary_op = tf.summary.merge_all()
    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)
    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
def main(argv=None):
  if gfile.Exists(FLAGS.eval_dir):
    gfile.DeleteRecursively(FLAGS.eval_dir)
  gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()
if __name__ == '__main__':
  tf.app.run()