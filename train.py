# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:08
# @Author  : jiaopan
# @Email   : jiaopaner@163.com
# 训练模块
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import network
import tfrecord
import utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', utils.configUtil("global.conf","model","model_dir"),
                           """模型保存目录"""
                           """检查点存储目录.(tensorboard查看)""")

tf.app.flags.DEFINE_integer('max_steps', int(utils.configUtil("global.conf","train","max_steps")),
                            """最大训练/迭代次数.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,"""""")

tf.app.flags.DEFINE_string('train_data',utils.configUtil("global.conf","train","train_tfrecord_dir"),
                           '训练集目录（tfrecord）')
tf.app.flags.DEFINE_integer('train_num',int(utils.configUtil("global.conf","train","train_data_count")),
                           '训练集样本总数')
def train():

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    # 获得图片数据和对应的标签batch
    float_image, label = tfrecord.train_data_read(tfrecord_path=FLAGS.train_data)
    images, labels = tfrecord.create_batch(float_image,label,count_num=FLAGS.train_num)

    logits = network.inference(images)
    # 误差计算
    loss = network.loss(logits, labels)
    # 模型训练
    train_op = network.train(loss, global_step)
    # 存储模型
    saver = tf.train.Saver(tf.global_variables())
    # 存储所有操作
    summary_op = tf.summary.merge_all()
    # 初始化所有变量.
    init = tf.initialize_all_variables()
    # 开始计算流图
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)
    # 队列开始
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                            graph_def=sess.graph_def)
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
      if step % 50 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
      # 保存模型检查点.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
def main(argv=None):  #命令行
  train()
if __name__ == '__main__':
  tf.app.run()