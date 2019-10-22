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
import copy

from os.path import dirname, abspath

import cv2
import os

from socket import *
from struct import pack, unpack
import numpy as np
import time


server_ip = '127.0.0.1'
server_port = 50000

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
        with socket(AF_INET, SOCK_STREAM) as s:
            s.bind((server_ip, server_port))
            s.listen(1)
            #(conn, addr) = s.accept()
            count = 0
#            try:
            while True:
                    
                    time.sleep(1.0)
    
                    count += 1
                    (conn, addr) = s.accept()
                    # receive data stream. it won't accept data packet greater than 1024 bytes
                    header = conn.recv(1)
                    images = []
                    
                    for i in range(2):
                        sb = conn.recv(4)
                        (length,) = unpack('I', sb)
                        #print(length)
                        
                        data_list = []
                        while len(data_list) < length:
                            #print(len(data_list))
                            to_read = length - len(data_list)
                            count +=1028
                            
                            data = conn.recv(
                                4096 if to_read > 4096 else to_read)
                            data_list.extend(data)
    
                        nparr = np.array(data_list, dtype=np.uint8)
                        nparr = nparr.reshape((2056, 2464))
                        images.append(nparr)
                        
                    
                    
                    if len(images) >= 2:
                        
                        detection = self.stereo_detection(
                                        images[0], img2 = images[1])
                    else:
                        detection = self.stereo_detection(
                                        images[0])
                    
                    #print("det2", detection)
                    
                
                    
                   # img1 = cv2.imread('cfg/practice_images/Manta1_mini/2018_10_17_12_58_10.52.jpg')
                    #img2 = cv2.imread('cfg/practice_images/Manta1_mini/2018_10_17_12_58_10.71.jpg')
                    #print(type(img1), type(img2))
                    #detection = self.stereo_detection(img1, img2)
                    if detection:
                        data = True
                    else:
                        data = False                   
                    #cv2.imshow("img1", img1)
                    #cv2.imshow("img2", img2)
                    #cv2.waitKey(0)
                    return_data = pack('?', data)
                    
                    conn.send(return_data)
                
                

            #finally:
                 #conn.shutdown(socket.SHUT_WR)
             #    conn.close()


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
