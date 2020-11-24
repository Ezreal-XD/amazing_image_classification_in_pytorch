import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import glob
import shutil


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def split_dataset():
    '''
    split pos232 into 3 part and generate corresponding txt file
    :return:
    '''
    dataset_dir = os.path.join("..", "data", "pos232")
    train_dir = os.path.join("..", "data", "train")
    val_dir = os.path.join("..", "data", "val")
    test_dir = os.path.join("..", "data", "test")

    makedir(train_dir)
    makedir(val_dir)
    makedir(test_dir)

    train_per = 0.8
    valid_per = 0.1
    test_per = 0.1

    imgs_list = []
    for img in os.listdir(dataset_dir):
        if 'mask' not in img:
            img_path = dataset_dir + '/' + img
            imgs_list.append(img_path)

    random.seed(666)
    random.shuffle(imgs_list)

    imgs_num = len(imgs_list)
    train_point = int(imgs_num * train_per)
    valid_point = int(imgs_num * (train_per + valid_per))

    for i in range(imgs_num):
        if i < train_point:
            shutil.copy(imgs_list[i], train_dir)
            shutil.copy(imgs_list[i].replace('.jpg', '_mask.jpg'), train_dir)
        elif i < valid_point:
            shutil.copy(imgs_list[i], val_dir)
            shutil.copy(imgs_list[i].replace('.jpg', '_mask.jpg'), val_dir)
        else:
            shutil.copy(imgs_list[i], test_dir)
            shutil.copy(imgs_list[i].replace('.jpg', '_mask.jpg'), test_dir)


def generate_txt():
    '''
    generate txt file for dataset
    '''

    train_txt_path = os.path.join("..", "data", "train.txt")
    val_txt_path = os.path.join("..", "data", "val.txt")
    test_txt_path = os.path.join("..", "data", "test.txt")

    def gen_txt(txt_path, dataset_type):
        f = open(txt_path, 'w')
        file_name = dataset_type + '.txt'
        dataset_dir = os.path.join("..", "data", dataset_type)

        for img in os.listdir(dataset_dir):
            if 'mask' not in img:
                img_path = dataset_type + '/' + img
                img_mask_path = dataset_type + '/' + img.replace('.jpg', '_mask.jpg')
                f.write(img_path + '\t' + img_mask_path + '\n')

        f.close()

    gen_txt(train_txt_path, 'train')
    gen_txt(val_txt_path, 'val')
    gen_txt(test_txt_path, 'test')


def save_predict(output, gt, img_name, dataset, save_path, output_grey=False, output_color=True, gt_color=False):
    if output_grey:
        output_grey = Image.fromarray(output)
        # output_grey.save(os.path.join(save_path, img_name + '.png'))

        f_path = os.path.join(save_path, img_name + '.png')
        f_path = f_path.replace('*', '$')
        output_grey.save(f_path)

    if output_color:
        if dataset == 'cityscapes':
            output_color = cityscapes_colorize_mask(output)
        elif dataset == 'camvid':
            output_color = camvid_colorize_mask(output)

        # output_color.save(os.path.join(save_path, img_name + '_color.png'))

        f_path = os.path.join(save_path, img_name + '.png')
        f_path = f_path.replace('*', '$')
        output_color.save(f_path)

    if gt_color:
        if dataset == 'cityscapes':
            gt_color = cityscapes_colorize_mask(gt)
        elif dataset == 'camvid':
            gt_color = camvid_colorize_mask(gt)

        gt_color.save(os.path.join(save_path, img_name + '_gt.png'))


def calParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


if __name__ == "__main__":
    # split_dataset()
    generate_txt()
