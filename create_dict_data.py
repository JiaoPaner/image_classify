# -*- coding:utf-8 -*-
import pickle
from PIL import Image
import numpy as np
import os
from struct import *
def read_file(filename,width=32,height=32):
    img_raw = Image.open(filename)
    img = img_raw.resize((width, height), Image.ANTIALIAS)
    r, g, b = img.split()
    r_array = np.array(r).reshape([1024])
    g_array = np.array(g).reshape([1024])
    b_array = np.array(b).reshape([1024])
    img_array = np.concatenate((r_array, g_array, b_array))
    return img_array

def image_input(rootDir):
    data = []
    labels = []
    batch_label = 'batch'
    filenames = []
    index = 0
    dirs = os.listdir(rootDir)
    for dir in dirs:
        for root, dirs, files in os.walk(rootDir + '/' + dir):
            for file in files:
                img_array = read_file(root + '/' + file)
                if data == []:
                    data = [img_array]
                else:
                    data = np.concatenate((data,[img_array]))
                print(data)
                labels.append(index)
                filenames.append(file)
                index = index+1

    return batch_label,labels,data,filenames

def pickle_save(batch_label,labels,data,filenames):
    print("正在存储")
    # 构造字典,所有的图像诗句都在arr数组里,我这里是个以为数组,目前并没有存label
    contact = {'batch_label':batch_label,'labels':labels,'data': data,'filenames':filenames}
    f = open('train_batch', 'wb+')
    pickle.dump(contact, f)#把字典存到文本中去
    f.close()
    print("存储完毕")
if __name__ == "__main__":
    #batch_label, labels, data, filenames = image_input('D:/learn/machine_learning/deep-learning/tensorflow/test')
    #pickle_save(batch_label,labels,data,filenames)
    #f = open('test_batch.bin', 'rb')
    #dic = pickle.load(f,encoding='bytes')
    #arr = dic['data']

    #print(dic)

    file = open("debug.txt", "wb+")
    file.write(pack("idh", 12345, 67.89, 2))
    file.close()
    file = open("debug.txt", "rb")

    print(unpack("idh", file.read(18)))
