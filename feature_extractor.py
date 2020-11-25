#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@author:fangpf
@time: 2020/11/24
"""
import argparse
import time

import cv2
import torch
import torchvision
from PIL import Image

import utils
from models.resnet import resnet50
from torchvision.transforms import transforms
import numpy as np

N_IDENTITY = 8631
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor()
     ]
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("face feature extractor")
    parser.add_argument('--weight', type=str, default='./models/resnet50_scratch_weight.pkl')
    parser.add_argument('--source', type=str, default='./')
    args = parser.parse_args()
    print(args)
    model = resnet50(num_classes=N_IDENTITY, include_top=False)
    utils.load_state_dict(model, args.weight)
    model = model.to(device)
    model.eval()
    image = args.source
    image = cv2.imread(image)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    tic = time.time()
    out = model(image)
    print(time.time() - tic)
    feature = out.view(out.shape[0], -1)
    print(feature.shape)
    feature = feature[0].data.cpu().numpy()
    feature_file = './feature/feature.npy'
    np.save(feature_file, feature)






