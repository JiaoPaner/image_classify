#coding:utf-8
# -*- coding: utf-8 -*-
# @Time    : 2018/4/11 10:08
# @Author  : jiaopan
# @Email   : jiaopaner@163.com

import tensorflow as tf
import os,sys,time
from PIL import Image
import utils

IMAGE_SIZE = int(utils.configUtil("global.conf","dataset","resize_image_size")) # 裁剪大小
IMAGE_MAT_SIZE = int(utils.configUtil("global.conf","dataset","image_mat_size")) # reshape参数
CHANNELS = int(utils.configUtil("global.conf","dataset","chnnels")) # 通道
TRAIN_DATASET = utils.configUtil("global.conf","dataset","train_data_dir") # 训练原始数据
EVAL_DATASET = utils.configUtil("global.conf","dataset","eval_data_dir") # 验证集原始数据
BATCH_SIZE = int(utils.configUtil("global.conf","dataset","batch_size"))

def create(dataset_dir,tfrecord_path,tfrecord_name="train_tfrecord",width=IMAGE_SIZE,height=IMAGE_SIZE):
    """
    #构建图片TFrecord文件
    param dataset_dir:原始图片的根目录,目录下包含多个子目录,每个子目录下为同一类别的图片
    param tfrecord_name:存储的TFreord文件名
    param tfrecord_path:存储的TFreord文件的目录路径
    param width:图片裁剪宽度
    param height:图片裁剪高度
    param channels:通道
    """
    if not os.path.exists(dataset_dir):
        print('创建TFRECORD文件时出错,文件目录或文件不存在,请检查路径名..\n')
        exit()
    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(os.path.dirname(tfrecord_path))
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path,tfrecord_name))
    lables = os.listdir(dataset_dir)
    print('一共 %d 个类别\n'% len(lables))
    for index,label in enumerate(lables):
        print('\n正在处理类别为 %s 的数据集' %label)
        time0 = time.time()
        filepath = os.path.join(dataset_dir,label)
        filesNames = os.listdir(filepath)
        for i,file in enumerate(filesNames):
            imgPath = os.path.join(filepath,file)
            img = Image.open(imgPath)
            img = img.resize((width,height))
            img = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>> Converting image %d/%d , %g s' % (
                i+1, len(filesNames), time.time() - time0))
    writer.close()
    print('\nFinished writing data to tfrecord files.')


def read(tfrecord_path,width,height,channels):
    """
    读取tfrecord 解析图片并预处理
    param tfrecord_path:tfrecord 文件目录
    param width,height:转换矩阵维度
    param channels:通道
    return: float_image,label
    """
    files = os.listdir(tfrecord_path)
    filenames = [os.path.join(tfrecord_path,tfrecord_name) for tfrecord_name in files]
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=True)
    reader = tf.TFRecordReader()
    key,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
                    "label": tf.FixedLenFeature([],tf.int64),
                    'image': tf.FixedLenFeature([],tf.string)
            })
    img = features['image'] #二进制数据
    img = tf.decode_raw(img,tf.uint8)
    img = tf.reshape(img,[channels,width,height])
    img = tf.transpose(img, [1, 2, 0]) # 转置原因:tf.nn.conv2d()的input参数shape为[batch, in_height, in_width, in_channels]
    label = tf.cast(features['label'], tf.int32)
    return img,label

def data_process(img):
    img = tf.cast(img, tf.float32)
    img = tf.image.random_flip_left_right(img)  # 图片随机左右翻转
    img = tf.image.random_brightness(img, max_delta=63)  # 变换图像的亮度
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)  # 变换图像的对比度
    float_image = tf.image.per_image_standardization(img)  # 图像标准化
    return float_image

def train_data_read(tfrecord_path,width=IMAGE_MAT_SIZE,height=IMAGE_MAT_SIZE,channels=CHANNELS):
    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)
        create(dataset_dir=TRAIN_DATASET,tfrecord_path=tfrecord_path)
    img, label = read(tfrecord_path, width, height, channels)
    #图片处理
    float_image = data_process(img)
    return float_image,label

def eval_data_read(tfrecord_path,width=IMAGE_MAT_SIZE,height=IMAGE_MAT_SIZE,channels=CHANNELS):
    if not os.path.exists(tfrecord_path):
        os.makedirs(tfrecord_path)
        create(dataset_dir=EVAL_DATASET,tfrecord_path=tfrecord_path)
    img, label = read(tfrecord_path, width, height, channels)
    img = tf.cast(img, tf.float32)
    float_image = tf.image.per_image_standardization(img)
    return float_image,label

#构建batch
def create_batch(float_image,label,count_num,batch_size=BATCH_SIZE):
    '''batch < min_after_dequeue < capacity'''
    capacity = int(count_num * 0.6 + 3 * BATCH_SIZE)
    min_after_dequeue = int(count_num * 0.6)
    images,label_batch = tf.train.shuffle_batch([float_image,label],batch_size=batch_size,
                                                capacity=capacity,min_after_dequeue=min_after_dequeue,num_threads=5)
    tf.summary.image('images', images)  # 图像可视化 tensorboard
    return images,label_batch

if __name__ == '__main__':

    #####测试#####

    #create(tfrecord_path='D:/learn/machine_learning/deep-learning/tensorflow/tfrecord/eval',
     #      tfrecord_name='tfrecord_eval',
      #     dataset_dir='D:/learn/machine_learning/deep-learning/tensorflow/test')
    float_image,label = train_data_read(tfrecord_path='D:/learn/machine_learning/deep-learning/tensorflow/tfrecord/train')
    images,label_batch = create_batch(float_image,label,1000)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(1):
            example, l = sess.run([images,label_batch])  # 在会话中取出image和label
            print(example, l)
        coord.request_stop()
        coord.join(threads)
