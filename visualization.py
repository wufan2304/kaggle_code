import numpy as np
import pandas as pd # 数据处理，CSV file I/O
import cv2
import matplotlib.pylab as plt
import os
from glob import glob

def plt_st(l1, l2):
    plt.figure(figsize=(l1, l2))

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
# from subprocess import check_output
# print(check_output(["dir", "../kaggle_code"]).decode("utf8"))
if __name__ == '__main__':
    TRAIN_DATA = "../train"
    type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*jpg"))
    type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1")) + 1: -4] for s in type_1_files])
    type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*jpg"))
    type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2")) + 1: -4] for s in type_2_files])
    type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*jpg"))
    type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3")) + 1: -4] for s in type_3_files])

    print(len(type_1_files), len(type_2_files), len(type_3_files))
    print("Type 1", type_1_ids[:10])
    print("Type 2", type_2_ids[:10])
    print("Type 3", type_3_ids[:10])

    TEST_DATA = "../test"
    test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
    test_ids = np.array([s[len(TEST_DATA) + 1: -4] for s in test_files])
    print(len(test_files))
    print(test_ids[:10])

    tile_size = (256, 256)
    n = 15

    complete_images = []

    for k, type_ids in enumerate([type_1_ids, type_2_ids, type_3_ids]):
        m = int(np.ceil(len(type_ids) * 1.0 / n))
        complete_image = np.zeros((m * (tile_size[0] + 2), n * (tile_size[1]+2), 3), dtype = np.uint8)
        train_ids = sorted(type_ids)
        counter = 0
        for i in range(m):
            ys = i * (tile_size[1] + 2)
            ye = ys + tile_size[1]
            for j in range(n):
                xs = j * (tile_size[0] + 2)
                xe = xs + tile_size[0]
                if counter == len(train_ids):
                    break
                image_id = train_ids[counter];counter += 1
                img = get_image_data(image_id, 'Type_%i' % (k+1))
                img = cv2.resize(img, dsize = tile_size)
                img = cv2.putText(img, image_id, (5, img.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness = 3)
                # plt.figure(counter)
                # plt.imshow(img)
                complete_image[ys:ye, xs:xe, :] =img[:, :, :]
            if counter == len(train_ids):
                break
        complete_images.append(complete_image)
        # plt_st(20,20)
        plt.figure(k)
        plt.imshow(complete_images[k])
        plt.title("Training dataset of type %i" % (k))
    plt.show()