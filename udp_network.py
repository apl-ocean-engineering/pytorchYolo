#!/usr/bin/env python3
"""
UDP interface with pytorchYolo

UDP data protocol is:
[size, image1_len, image1, image2_len, image2,...,imageN_len, imageN]
Where:
    size: a 4 bit int corresponding to number of images in package
    image1_len: 8 bit unsigned long of number of bytes in image1
    image1: raw image data

    Repeate for N images. Currently supports only 2
"""

from pytorchYolo.detector import YoloLiveVideoStream
from pytorchYolo.argLoader import ArgLoader
from ampProc.stereo_processing import StereoProcessing

from os.path import dirname, abspath

import cv2
import os

from socket import *
from struct import pack, unpack
import numpy as np
import time


server_ip = '127.0.0.1'
server_port = 50001

IMG1_NAME='img1'
IMG2_NAME='img2'

#TODO: Temp moving here for practice return...
class Square:
    def __init__(self, lower_corner, width, height):
        lower_x = int(lower_corner[0])
        lower_y = int(lower_corner[1])
        self.upper_x = int(lower_x + width)
        self.upper_y = int(lower_y + height)

        self.lower_x = max(0, lower_x)
        self.lower_y = max(0, lower_y)

    def print_square(self):
        print((self.lower_x, self.lower_y), (self.upper_x, self.upper_y))

SQUARE_LIST = [Square((0,0), 10, 10), Square((0,0), 10, 10),Square((0,0), 10, 10)]
CLASS_LIST = ['FISH', 'FISH', 'FISH']

class ServerProtocol:

    def __init__(self, args, detector):
        self.args = args
        self.SP = StereoProcessing(args, detector)

        self.socket = None
        self.output_dir = '.'
        self.file_num = 1

        self.detector = detector

        cv2.namedWindow(IMG1_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(IMG2_NAME, cv2.WINDOW_NORMAL)


    def handle_images(self, server_ip, server_port):
        """
        Accepts 2 MANTA Images
        """
        with socket(AF_INET, SOCK_STREAM) as s:
            s.bind((server_ip, server_port))
            s.listen(1)
            (conn, addr) = s.accept()
            if self.args.TCP_string:
                buf = b''
            while True:
                    time_inital = time.time()
                    header = conn.recv(1)
                    images = []
                    for i in range(2):
                        count = 0
                        #print('here')
                        sb = conn.recv(4)
                        (length,) = unpack('I', sb)
                        #print(length)
                        data_arr_lst = []
                        while count < length:
                            to_read = length - count
                            data = conn.recv(
                                4096 if to_read > 4096 else to_read)
                            if self.args.TCP_string:
                                buf += data
                            else:
                                buff = np.frombuffer(data, np.uint8)
                                data_arr_lst.append(buff)
                            count+=len(data)
                        if self.args.TCP_string:
                            img = np.fromstring(buf, dtype='uint8').reshape((self.args.height,
                                             self.args.width))
                        else:
                            img = np.concatenate(data_arr_lst,
                                  axis=0).reshape((self.args.height,
                                                   self.args.width))
                        #print(img.shape)
                        images.append(img)

                    cv2.imshow(IMG1_NAME, images[0])

                    if self.args.show_image:
                        cv2.waitKey(1)
                    if len(images) >= 2:
                        detection, squares1, squares2, class_list1, class_list2 \
                            = self.stereo_detection(
                                        images[0], img2 = images[1])
                    else:
                        detection, squares1, squares2, class_list1, class_list2 \
                            = self.stereo_detection(
                                        images[0])

                    detection = True

                    """
                    Return data:
                        1. bool indicating true/false
                        if True:
                            2. Length of detections (i.e. bounding boxes)
                            3. 'N' Bounding boxes packed as 4 ints
                            4. 'N' Detection names as ints
                    """ 
                    return_data = pack('?', detection)
                    conn.send(return_data)

                    if detection:
                        for i in range(2):
                            detection_length = pack('I', len(SQUARE_LIST))
                            conn.send(detection_length)
                            for sq in SQUARE_LIST:
                                square_information = [sq.lower_x, sq.lower_y,
                                    sq.upper_x, sq.upper_y]
                                packed_sq = pack('%sf' % len(square_information),
                                            *square_information)
                                conn.send(packed_sq)
                            for class_name in CLASS_LIST:
                                packed_class = pack('I', 1)
                                conn.send(packed_class)









    def stereo_detection(self, img1, img2 = None):
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        self.detector.display = False
        detection, squares1, squares2, class_list1, class_list2 = \
            self.SP.run_images(img1, img2=img2)

        return detection, squares1, squares2, class_list1, class_list2


    def close(self):
        print("close")
        self.socket.close()
        self.socket = None

if __name__ == '__main__':
    argLoader = ArgLoader()
    argLoader.parser.add_argument('--show_image', help="Show images",
        default=True)
    argLoader.parser.add_argument('--width', help="img width", default=2464, type=int)
    argLoader.parser.add_argument('--height', help="img height", default=2056, type=int)
    argLoader.parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(
            __file__))) + "/cfg/calibrationConfig.yaml")
    argLoader.parser.add_argument(
        "--base_path", help="Base folder to calibration values",
        default=dirname(dirname(abspath(__file__))) + "/calibration/")
    argLoader.parser.add_argument(
        "--TCP_string", default=False)

    args = argLoader.args  # parse the command line arguments

    detector = YoloLiveVideoStream(args)


    sp = ServerProtocol(args, detector)
    sp.handle_images(server_ip, server_port)
