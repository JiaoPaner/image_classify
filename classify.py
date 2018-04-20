# -*- coding: utf-8 -*-
# @Time    : 2018/4/15 2:14
# @Author  : jiaopan
# @Email   : jiaopaner@163.com
from PIL import Image
import tensorflow as tf
import network
import utils
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', utils.configUtil("global.conf","model","model_dir"),
                           """保存的模型.""")
def inputs(input,count=1,batch_size=1):
    network.FLAGS.batch_size = batch_size
    img = Image.open(input)
    img = img.resize((32, 32))
    img = img.tobytes()
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [3, 32, 32])
    img = tf.transpose(img, [1, 2, 0])
    img = tf.cast(img, tf.float32)
    float_image = tf.image.per_image_standardization(img)
    capacity = int(count * 0.4 + 3 * batch_size)
    min_after_dequeue = int(batch_size * 0.4)
    images, label_batch = tf.train.shuffle_batch([float_image, '?'], batch_size=batch_size,
                                                 capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=5)
    return images
def predict(imgPath):
    with tf.Graph().as_default():
        float_image = inputs(imgPath)
        logits = network.inference(float_image)
        top_k_op = tf.nn.in_top_k(logits, [1], 1)
        variable_averages = tf.train.ExponentialMovingAverage(network.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter('/tmp/jiaopan/pred',
                                               graph_def=graph_def)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                predictions = sess.run([logits])  # e.g. return [true,false,true,false,false]
                print(predictions)
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=1)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:
                coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)

def main(argv=None):
    predict("D:/learn/machine_learning/deep-learning/tensorflow/faces/jp/33.bmp")
if __name__ == '__main__':
    tf.app.run()
