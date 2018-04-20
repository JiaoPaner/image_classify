# image_classify
tensorflow实现卷积神经网络并训练自定义数据集
整个模型建立流程如下:

1.执行tfrecord.py生成tfrecord训练集和验证集文件,训练集合验证集生成时参数设置应一致

2.执行train.py开始训练模型 每训练1000次后保存或更新模型

3.执行train.py的同时执行network_eval.py它会在设定的时间间隔内,加载保存的模型,利用验证数据集去评估模型的准确率

4.执行classify.py对新图片进行分类

