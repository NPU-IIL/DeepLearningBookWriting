"""设置数据集"""
from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import os
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve

"""设置文件下载路径和下载文件"""
def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)

def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/' # MNIST数据集的基本URL

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh: # 使用gzip打开压缩文件
            magic, num_data = struct.unpack(">II", fh.read(8)) # 读取文件头信息
            return np.array(array.array("B", fh.read()), dtype=np.uint8) # 解析标签数据并返回NumPy数组

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:  # 使用gzip打开压缩文件
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))  # 读取文件头信息
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)  # 解析图像数据并返回NumPy数组

    for filename in ['train-images-idx3-ubyte.gz',  # 需要下载和解析的文件名列表
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)  # 下载文件

    train_images = parse_images('data/train-images-idx3-ubyte.gz')  # 解析训练图像数据
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')  # 解析训练标签数据
    test_images  = parse_images('data/t10k-images-idx3-ubyte.gz')  # 解析测试图像数据
    test_labels  = parse_labels('data/t10k-labels-idx1-ubyte.gz')  # 解析测试标签数据

    return train_images, train_labels, test_images, test_labels  # 返回解析后的数据
