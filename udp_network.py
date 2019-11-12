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
server_port = 50000

IMG1_NAME='img1'
IMG2_NAME='img2'

class ServerProtocol:

    def __init__(self, args, detector):
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

            while True:
                    time_inital = time.time()
                    header = conn.recv(1)
                    images = []
                    for i in range(2):
                        count = 0
                        sb = conn.recv(4)
                        (length,) = unpack('I', sb)
                        data_arr_lst = []
                        while count < length:
                            to_read = length - count

                            data = conn.recv(
                                4096 if to_read > 4096 else to_read)

                            buff = np.frombuffer(data, np.uint8)
                            data_arr_lst.append(buff)
                            count+=len(data)
                        img = np.concatenate(data_arr_lst, axis=0).reshape((2056, 2464))
                        images.append(img)

                    print("Time elapsed", time.time() - time_inital)
                    cv2.imshow(IMG1_NAME, images[0])
                    if len(images) >= 2:

                        detection = self.stereo_detection(
                                        images[0], img2 = images[1])
                    else:
                        detection = self.stereo_detection(
                                        images[0])


                    detection = True
                    if detection:
                        data = True
                    else:
                        data = False

                    return_data = pack('?', data)



                    conn.send(return_data)



    def stereo_detection(self, img1, img2 = None):
        #if img1.shape[2] != 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        self.detector.display = False
        detection = self.SP.run_images(img1, img2=img2)

        return detection


    def close(self):
        print("close")
        self.socket.close()
        self.socket = None

if __name__ == '__main__':
    argLoader = ArgLoader()
    argLoader.parser.add_argument(
        "--calibration_yaml",
        help="Path to calibration yaml specify path of calibration files",
        default=dirname(dirname(abspath(
            __file__))) + "/cfg/calibrationConfig.yaml")
    argLoader.parser.add_argument(
        "--base_path", help="Base folder to calibration values",
        default=dirname(dirname(abspath(__file__))) + "/calibration/")

    args = argLoader.args  # parse the command line arguments

    detector = YoloLiveVideoStream(args)

    sp = ServerProtocol(args, detector)
    sp.handle_images(server_ip, server_port)
