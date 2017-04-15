# 由于存储的文件太大，加一起有200多M，在效率上会造成太大损失，所以这个文件就算了。


from glob import glob
import os
import numpy as np
import cv2
import sys,time
# from PIL import Image
import csv

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
TRAIN_DATA = "../train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1")) + 1: -4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2")) + 1: -4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3")) + 1: -4] for s in type_3_files])
TEST_DATA = "../test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA) + 1: -4] for s in test_files])

tile_size = (128,128)

train_image = []
for k, type_ids in enumerate([type_1_ids,type_2_ids,type_3_ids]):
    file_name = "Type_%i" % (k + 1) + ".csv"
    with open(file_name,'w') as datacsv:
        csv_writer = csv.writer(datacsv)
        for i in range(5):
            image_id = type_ids[i]
            img = get_image_data(image_id, 'Type_%i' % (k + 1))
            img = cv2.resize(img, dsize=tile_size)
            temp =img.reshape((1,-1))
            if train_image!=[]:
                train_image = np.row_stack((train_image,temp))
            else:
                train_image = temp

            count = int(i *100 / len(type_ids))
            sys.stdout.write('\r File resizing progress: %2d%% completed!'%count)
            sys.stdout.flush()
