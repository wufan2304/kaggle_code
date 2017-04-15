import random
import numpy as np
from glob import glob
import os


TRAIN_DATA_PATH = "../train_resize"
image_files = glob(os.path.join(TRAIN_DATA_PATH, "*jpg"))
__image_ids = np.array([s[len(os.path.join(TRAIN_DATA_PATH)) + 1: -4] for s in image_files])


def split_train_data(image_ids = __image_ids ,test_size = 0.2):
    X_train_ids = []
    X_test_ids = []
    b = ['5','4','6','8','9','6','3']
    # print(b)
    a = random.shuffle(b)
    print(a)
    # for k, image_id in enumerate(image_ids):


    return X_train_ids,X_test_ids

if __name__ == '__main__':
    x,y = split_train_data(__image_ids,0.2)
