#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:14:00 2019

@author: mitchell
"""
import cv2
#img_width = 2464
#img_width = 2056
#img_width = 1280
#img_height = 960

train_cfg_name = "/home/mitchell/YOLO_ws/pytorchYolo/cfg/train_test_locs/WAMP_updated_training/WAMP_retrain_train.txt"
#train_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/train_all_data.txt"
#train_cfg_name = "/home/mitchell/pytorchYolo/cfg/train_test_locs/train_eyesea_wells.txt"

with open(train_cfg_name) as f:
    for line in f:

        img = cv2.imread(line.rstrip())
        img_width = img.shape[1]
        img_height = img.shape[0]

        label = str(line.replace('images', 'labels')).replace('jpg', 'txt')

        if True:
            split_label = label.split('/')
            new_name = split_label[3].replace('labels', 'images')
            split_label[3] = new_name
            label = '/'.join(split_label)

        with open(label.rstrip()) as label:
            for _bbox in label:

                bbox = _bbox.split(' ')
                x_min = float(bbox[1])
                y_min = float(bbox[2])
                box_width = float(bbox[3])
                box_height = float(bbox[4])
               # print(x_min, y_min, box_width, box_height)

                #bottom_loc = (int(x_min*img_height), int((y_min-box_height)*img_width))
                #top_loc = (int((x_min+box_width)*img_width), (int(y_min*img_height)))
                #print('bottom', bottom_loc)
                #print('top', top_loc)
                #cv2.rectangle(img, bottom_loc, top_loc, (255,0,0), 2)
                x = int(float(x_min - box_width/2)*img_width)
                y = int(float(y_min - box_height/2)*img_height)
                left_top = (x, y)
                x2 = int(float(x_min + box_width/2)*img_width)
                y2 = int(float(y_min + box_height/2)*img_height)
                bottom_right = (x2, y2)

                center = (int(x_min*img_width), int(y_min*img_height))
                cv2.circle(img, left_top, 10, (255,0,255))
                cv2.circle(img, center, 10, (0,255,0))
                cv2.rectangle(img, left_top, bottom_right, (255,0,0), 2)
        print(img.shape[0] > 60)
        if (img.shape[0] > 60):
            cv2.imshow("img", img)
            k = cv2.waitKey(100)
