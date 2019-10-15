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
from pytorchYolo.stereo_processing import StereoProcessing

from os.path import dirname, abspath

import cv2
import os

from socket import *
from struct import pack, unpack
import numpy as np
import time


IP = '127.0.0.1'
port = 5000

class ServerProtocol:

    def __init__(self, args, detector):
        self.SP = StereoProcessing(args, detector)

        self.socket = None
        self.output_dir = '.'
        self.file_num = 1

        self.detector = detector

    def handle_images(self, server_ip, server_port):
        """
        Accepts 2 MANTA Images
        """
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(5)
        (conn, addr) = self.socket.accept()
        try:
            while True:
                # receive data stream. it won't accept data packet greater than 1024 bytes
                header = conn.recv(1)
                sb = conn.recv(4)
                (size,) = unpack('!I', sb)
                images = []
                for i in range(size):
                    bs = conn.recv(8)
                    (length,) = unpack('>Q', bs)
                    data = b''
                    while len(data) < length:
                        to_read = length - len(data)
                        data += conn.recv(
                            4096 if to_read > 4096 else to_read)

                    nparr = np.fromstring(data, np.uint8)
                    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(img_np)
                if len(images) >= 2:
                    detection = self.stereo_detection(
                                    images[0], img2 = images[1])
                else:
                    detection = self.stereo_detection(
                                    images[0])

                if detection:
                    data = True
                else:
                    data = False
                return_data = pack('?', data)
                conn.send(return_data)

        finally:
             #conn.shutdown(socket.SHUT_WR)
             conn.close()


    def stereo_detection(self, img1, img2 = None):
        # time_init = time.time()
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
    sp.handle_images(IP, port)
