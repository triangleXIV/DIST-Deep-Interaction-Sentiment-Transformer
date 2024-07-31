# -*- coding: utf-8 -*-
# file: data_utils_ai.py
# author: xunan <xunan0812@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
import pickle
import os
from PIL import Image
from torchvision import transforms
from collections import Counter
from torch.utils.data import Dataset
import torch
import numpy as np
from swin.swintransformer import SwinTransformer,get_config
from yacs.config import CfgNode as CN
from transformers import BertTokenizer
from logs import logger

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_C = CN()
_C.BASE = ['']
# Model settings
_C.MODEL = CN()
_C.MODEL.NAME = 'swin_small'
_C.MODEL.PATCH_SIZE = 16
_C.MODEL.EMBED_DIMS = 384
_C.MODEL.DEPTH = 24
_C.MODEL.NUM_HEADS = 8
_C.MODEL.INIT_VALUES = 1e-5
_C.MODEL.IN_CHANS = 3
_C.MODEL.DROP_RATE = 0.1
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.IMAGE_SIZE = 224

config = _C.clone()

class ZOLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ZOLDatesetReader:
    @staticmethod
    def __data_Counter__(fnames):
        jieba_counter = Counter()
        label_counter = Counter()
        max_length_text = 0
        min_length_text = 1000
        max_length_img = 0
        min_length_img = 1000
        lengths_text = []
        lengths_img = []
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 4):
                    text_raw = lines[i].strip()
                    imgs = lines[i + 1].strip()[1:-1].split(',')
                    aspect = lines[i + 2].strip()
                    polarity = lines[i + 3].strip()

                    length_text = len(text_raw)
                    length_img = len(imgs)

                    if length_text >= max_length_text:
                        max_length_text = length_text
                    if (length_text <= min_length_text):
                        min_length_text = length_text
                    lengths_text.append(length_text)

                    if length_img >= max_length_img:
                        max_length_img = length_img
                    if (length_img <= min_length_img):
                        min_length_img = length_img
                    lengths_img.append(length_img)
                    jieba_counter.update(text_raw)
                    label_counter.update([polarity])
        print(
            'data_num:', len(lengths_text),
            'max_length_text:', max_length_text,
            'min_length_text:', min_length_text,
            'ave_length_test:', np.average(np.array(lengths_text)),
            'max_length_img:', max_length_img,
            'min_length_img:', min_length_img,
            'ave_length_img:', np.average(np.array(lengths_img)),
            'jieba_num:', len(jieba_counter)
        )
        print(label_counter)

        # data_num: 28429
        # max_length_text: 8511
        # min_length_text: 5
        # ave_length_test: 315.106651659
        # max_length_img: 111
        # min_length_img: 1
        # ave_length_img: 4.49984171093
        # jieba_num: 3389

        # data_num: 28429
        # max_length_text: 8511
        # min_length_text: 5
        # ave_length_text: 315.106651659
        # max_length_img: 111
        # min_length_img: 1
        # ave_length_img: 4.49984171093
        # jieba_num: 3389

    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 4):
                    text_raw = lines[i].strip()
                    text += text_raw + " "
        return text

    def read_img(self, imgs_path):
        imgs = []
        num = 0
        loaded_imgs = []
        imgs_path=[imgs_path]
        # 读取并存储图片
        for j in range(len(imgs_path)):
            if num == self.max_img_len:
                break
            img_path = imgs_path[j].strip().replace('\'', '').replace('"', '')
            try:
                img = Image.open('./datasets/img/' + img_path).convert('RGB')
                loaded_imgs.append(img)
                num += 1
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")  # 打印错误信息

        if not loaded_imgs:
            print("No images were successfully loaded.")

        # 对所有成功读取的图片进行特征提取
        with torch.no_grad():
            inputs = [self.transform(img).unsqueeze(0).to('cuda') for img in loaded_imgs]
            if inputs:
                inputs = torch.cat(inputs)  # 合并所有输入
                outputs = self.feature_extractor(inputs)

                for i in range(outputs.size(0)):
                    imgs.append(outputs[i])

        embed_dim_img = len(imgs[0])
        img_features = torch.zeros(self.max_img_len, embed_dim_img)
        num_imgs = len(imgs)
        if num_imgs >= self.max_img_len:
            for i in range(self.max_img_len):
                img_features[i, :] = imgs[i]
        else:
            for i in range(self.max_img_len):
                if i < num_imgs:
                    # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                    img_features[i, :] = imgs[i]
                else:
                    break
        return img_features, min(self.max_img_len, num_imgs)

    # @staticmethod
    def read_data(self, fname, tokenizer):
        polarity_dic = {'-1': 0, '0': 1, '1': 2}
        data_path = fname.split('.txt')[0]+'/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        data_path = fname.split('.txt')[0]+'/'+self.model_name+'/'
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            lines = fin.readlines()
            all_data = []
            for i in range(0, len(lines), 4):
                fname_i = data_path + str(int(i/4)) + '.pkl'
                if os.path.exists(fname_i):
                    with open(fname_i, 'rb') as fpkl:
                        data = pickle.load(fpkl)
                else:
                    logger.info(fname_i)
                    text_raw = lines[i].strip()
                    imgs, num_imgs = self.read_img(lines[i + 3].strip())# 图像特征提取 这部分不参与训练
                    aspect = lines[i + 1].strip()#获取评价的方面 如续航
                    polarity = int(polarity_dic[(lines[i + 2].strip())])#获取评分 转化为对应分类编号 8分类
                    text_raw_indices = tokenizer(text_raw, padding='max_length', truncation=True, max_length=self.max_seq_len, return_tensors="pt")
                    aspect_indices = tokenizer(aspect, padding='max_length', truncation=True, max_length=self.max_aspect_len, return_tensors="pt")
                    text_raw_indices_mask=text_raw_indices['attention_mask'].squeeze(0)
                    text_raw_indices_token = torch.zeros((1, 64), dtype=torch.int).squeeze(0)
                    #text_raw_indices_token=text_raw_indices['token_type_ids'].squeeze(0)

                    text_raw_indices=text_raw_indices['input_ids'].squeeze(0)
                    aspect_indices_mask = aspect_indices['attention_mask'].squeeze(0)
                    #aspect_indices_token = aspect_indices['token_type_ids'].squeeze(0)
                    aspect_indices_token = torch.zeros((1, 16), dtype=torch.int).squeeze(0)
                    aspect_indices = aspect_indices['input_ids'].squeeze(0)

                    img_mask = [1] * num_imgs + [0] * (self.max_img_len - num_imgs)
                    img_mask =  torch.tensor(img_mask, dtype=torch.int64)
                    data = {
                        'text_raw_indices': text_raw_indices, # 一句话对应id
                        'text_raw_indices_token': text_raw_indices_token,
                        'text_raw_indices_mask': text_raw_indices_mask,
                        'aspect_indices': aspect_indices,  # 方面id
                        'aspect_indices_token': aspect_indices_token,
                        'aspect_indices_mask': aspect_indices_mask,
                        'imgs': imgs, # n张 dims的向量作为类似token的存在
                        'num_imgs': num_imgs,
                        'img_mask': img_mask,
                        'polarity': int(polarity),#评分 即 label
                    }
                    with open(fname_i, 'wb') as fpkl:#保存一组评价 4行为一组
                        pickle.dump(data, fpkl)
                all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', max_seq_len=300, max_aspect_len=6, max_img_len=5,img_size=384):
        start = time.time()
        print("Preparing {0} datasets...".format(dataset))
        fname = {
            'zol_cellphone': {
                'train': './datasets/train.txt',
                'dev': './datasets/dev.txt',
                'test': './datasets/test.txt'
            }
        }

        self.model_name = "swin"
        self.max_img_len = max_img_len
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        config = get_config('./pretrains/swin_large_patch4_window12_384_22kto1k_finetune.yaml')
        self.feature_extractor = SwinTransformer(img_size=384,
                              patch_size=config.MODEL.SWIN.PATCH_SIZE,
                              in_chans=config.MODEL.SWIN.IN_CHANS,
                              num_classes=config.MODEL.NUM_CLASSES,
                              embed_dim=config.MODEL.SWIN.EMBED_DIM,
                              depths=config.MODEL.SWIN.DEPTHS,
                              num_heads=config.MODEL.SWIN.NUM_HEADS,
                              window_size=config.MODEL.SWIN.WINDOW_SIZE,
                              mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                              qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                              qk_scale=config.MODEL.SWIN.QK_SCALE,
                              drop_rate=config.MODEL.DROP_RATE,
                              drop_path_rate=config.MODEL.DROP_PATH_RATE,
                              patch_norm=config.MODEL.SWIN.PATCH_NORM,
                              use_checkpoint=False)


        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=3),  # 调整图像大小
            transforms.CenterCrop(img_size),  # 中心裁剪
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pretrained_dict = torch.load('./pretrains/swin_large_patch4_window12_384_22kto1k.pth', map_location='cpu')
        pretrained_dict = pretrained_dict['model']
        unexpected_keys = {"head.weight", "head.bias"}
        for key in unexpected_keys:
            del pretrained_dict[key]
        missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(pretrained_dict, strict=False)

        tokenizer = AutoTokenizer.from_pretrained('./pretrains')
        self.feature_extractor.to('cuda')
        self.train_data = ZOLDataset(self.read_data(fname[dataset]['train'], tokenizer))
        self.dev_data = ZOLDataset(self.read_data(fname[dataset]['dev'], tokenizer))
        self.test_data = ZOLDataset(self.read_data(fname[dataset]['test'], tokenizer))
        end = time.time()
        m, s = divmod(end-start, 60)
        print('Time to read datasets: %02d:%02d' % (m, s))
        torch.cuda.empty_cache()

if __name__ == '__main__':

    # text_zol = ZOLDatesetReader.__read_text__(['./datasets/zolDataset/zol_Train_jieba.txt',
    #                                            './datasets/zolDataset/zol_Dev_jieba.txt',
    #                                            './datasets/zolDataset/zol_Test_jieba.txt'])
    # counter_zol = ZOLDatesetReader.__data_Counter__(['./datasets/zolDataset/zol_Train_jieba.txt',
    #                                            './datasets/zolDataset/zol_Dev_jieba.txt',
    #                                            './datasets/zolDataset/zol_Test_jieba.txt'])
    zol_dataset = ZOLDatesetReader()

