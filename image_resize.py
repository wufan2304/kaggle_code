from glob import glob
import os
import numpy as np
import cv2
import sys,time
# from PIL import Image
import csv


# 该文件是最原始图像进行resize操作，图像resize的大小由tile_size来控制，即
# tile_size = (128, 128)指的是将图片resize为128*128大小
# 输出图片的存储地址为 Image_output_dir

# 训练数据集存放位置
TRAIN_DATA = "../train"
# 测试数据集存放位置
TEST_DATA = "../test"
# 图片resize的目标大小
tile_size = (64,64)
# 图片的存储地址，例如  “上一层目录中的 / image_resize_128 中”
Image_output_dir = "../image_resize_" + str(tile_size[0])
# 判断文件目录是否存在，如果不存在，就创建目录；如果存在，则跳过
if os.path.exists(Image_output_dir)==False:
    os.makedirs(Image_output_dir)


def get_filename(image_id, image_type):
    if image_type == "Type_1" or image_type == "Type_2" or image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or image_type == "AType_2" or image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)
    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def get_image_data(image_id, image_type):
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# 获得train和test的所有的图片id
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1")) + 1: -4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2")) + 1: -4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3")) + 1: -4] for s in type_3_files])
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA) + 1: -4] for s in test_files])


# 将图片resize后存放在目标文件夹中
# 目标文件夹为上一层目录中的Image_output_dir
# 存放并没有按照type类型分开存放
for k, type_ids in enumerate([type_1_ids,type_2_ids,type_3_ids]):
    # print(list(range(len(type_ids))))
    for i in range(len(type_ids)):
        image_id = type_ids[i]
        img = get_image_data(image_id, 'Type_%i' % (k + 1))
        img = cv2.resize(img, dsize=tile_size)
        # print(img.reshape([1,-1,-1,-1]))
        filename = Image_output_dir + "/" + str(image_id)+".jpg"
        # print("total number = ", len(type_ids), " i = ", i , " image_id = " + image_id, ", k = ", k)
        cv2.imwrite(filename,img)
        # 下面的跟图片处理无关，仅仅用于进度显示
        count = int(i *100 / len(type_ids))
        sys.stdout.write('\r File resizing progress: %2d%% complete!'%count)
        sys.stdout.flush()




# 下面对train图片的label进行统计
with open(Image_output_dir + '/train.csv',"w",newline="") as datacsv:
    csvwriter = csv.writer(datacsv, dialect=("excel"))
    csvwriter.writerow(['filename', 'image_label'])
    for k, type_ids in enumerate([type_1_ids,type_2_ids,type_3_ids]):
        # print(list(range(len(type_ids))))
        for i in range(len(type_ids)):
            image_id = type_ids[i]
            filename = str(image_id) + ".jpg"
            image_label = k+1
            csvwriter.writerow([filename,image_label])
            # 下面的语句与执行任务无关，只用于进度显示
            count = int(i *100 / len(type_ids))
            sys.stdout.write('\r File resizing progress: %2d%% complete!'%count)
            sys.stdout.flush()
