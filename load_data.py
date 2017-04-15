from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import cv2,os,random,sys,csv
import numpy as np
import pandas as pd
from glob import glob
from convert_types_into_seq import convert_types_into_seq





# 处理好的128*128大小的图存储地址
TRAIN_DATA_PATH = "../train_resize_all_types_size_is_128"
# 获取所有图的地址+名字
image_files = glob(os.path.join(TRAIN_DATA_PATH, "*jpg"))
# 提取所有图的名字
X_train_filename = np.array([s[len(os.path.join(TRAIN_DATA_PATH)) + 1: ] for s in image_files])
# 提取所有图的编号
image_ids = np.array([s[len(os.path.join(TRAIN_DATA_PATH)) + 1: -4] for s in image_files])
# 对图的编号进行顺序打乱
random.shuffle(image_ids)

# 将编号分为两组，一组作为训练数据，另一组作为测试数据
# 设定测试数据所占比例
percentage = 0.2
# 计算训练数据的大小
train_size = int((1-percentage) * len(image_ids))
# 提取打乱后的编号中，前train_size个，作为训练数据的id号
X_train_ids = image_ids[:train_size]
# 提取打乱后的编号中，后面的部分，作为测试数据的id号
X_test_ids = image_ids[train_size:]

# 读取label文件，格式如下
#           'filename'      'image_label'
# 0         '0.jpg'         1
# 1         '10.jpg'        2
# 2         '50.jpg'        3
# 3         '121.jpg'       1
train_label = pd.read_csv('train.csv')

# 根据设定好的分组id号，分别将训练数据和测试数据读入到返回值中
# 输入为id号的列表，输出为图的行向量表，和label表
def load_image(ids):
    result = np.array('')
    label = np.array('')
    # 对每一幅图的id进行遍历
    for i, image_id in enumerate(ids):

        # 获取这幅图的数据，存入temp
        # 将得到的id号转化为这幅图的具体名称地址
        image_name =TRAIN_DATA_PATH + '/' + str(image_id) + '.jpg'
        # 读入这幅图
        temp = cv2.imread(image_name)
        # 将128*128*3大小的图转换为行向量，类型为numpy.ndarray
        temp = temp.reshape((1,-1))[0,:128*128]
        temp = temp / np.max(temp)
        # print(temp/np.max(temp))
        # print(temp)
        # 查询这幅图对应的label，类型为numpy.ndarray，存入temp_label
        temp_label = train_label[train_label['filename']==str(image_id)+'.jpg'].values

        # 将temp和temp_label的数据，按照新增加一行的形式，分别添加到result和label的末尾
        if i==0:
            result = temp
            label = temp_label
        else:
            result = np.row_stack((result,temp))
            label = np.row_stack((label,temp_label))

        # 处理进度显示，百分比 %
        # 计算百分比
        count = int(i *100 / len(ids))
        # 显示百分比
        sys.stdout.write('\r image loading progress: %2d%% completed!'%count)
        # 消除上一步的百分比显示，等待下一次的显示
        sys.stdout.flush()
    return [result,label]


# 主程序部分
# 根据id号的分配，完成数据和label的读入
X_train,Y_train = load_image(X_train_ids)
X_test,Y_test = load_image(X_test_ids)

X_train = X_train.reshape((-1,128,128,1))
X_test = X_test.reshape((-1,128,128,1))
Y_train = convert_types_into_seq(Y_train)
Y_test = convert_types_into_seq(Y_test)


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=90.)


# Building convolutional network
network = input_data(shape=[None, 128, 128, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X_train}, {'target': Y_train}, n_epoch=20,
           validation_set=({'input': X_test}, {'target': Y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')


