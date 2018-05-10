import random
from PIL import Image
import numpy as np
import os
import h5py

IMAGE_DIR = ['../face_into/image_face', '../face_into/image_face']
HDF5_FILE = ['hdf5_train.h5', 'hdf5_test.h5']
LIST_FILE = ['face6.txt', 'face6.txt']
LABELS = dict(
    # (kind_1, kind_2)
    A_0=(0, 0),
    B_0=(1, 0),
    A_1=(0, 1),
    B_1=(1, 1),
    A_2=(0, 2),
    B_2=(1, 2),
)

for kk, image_dir in enumerate(IMAGE_DIR):
    print('kk, image_dir',kk, image_dir)
    file_list = ...
    random.shuffle(file_list)

    kind_index = ...
    datas = np.zeros((len(file_list), 1, 32, 96))
    labels = np.zeros((len(file_list), 2))

    for ii, _file in enumerate(file_list):
        # hdf5文件要求数据是float或者double格式
        # 同时caffe中Hdf5DataLayer不允许使用transform_param，
        # 所以要手动除以256
        datas[ii, :, :, :] = \
            np.array(Image.open(_file)).astype(np.float32) / 256
        labels[ii, :] = np.array(LABELS[kind_index]).astype(np.int)

        # 写入hdf5文件
    with h5py.File(HDF5_FILE[kk], 'w') as f:
        f['data'] = datas
        f['labels'] = labels
        f.close()

    with open(LIST_FILE[kk], 'w') as f:
        f.write(os.path.abspath(HDF5_FILE[kk]) + '\n')
        f.close()