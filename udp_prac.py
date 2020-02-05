# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:55:08 2019

@author: AMP
"""

import socket
from struct import unpack
import numpy as np
import cv2
import sys
import time

IP = '127.0.0.1'
port = 50000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket:

    socket.bind((IP, port))
    socket.listen(1)
    
    while True:
        (conn, addr) = socket.accept()
        header = conn.recv(1)
        print(header)
        #print(header)
        """
        packet_num_sb = conn.recv(2)
        
        (packet_size,) = unpack('>h', packet_num_sb)
        """
        #print(packet_size)
        
        #for i in range(packet_size):
        bs = conn.recv(4)
        #print(int(bs, 16))
        (length,) = unpack('I', bs) 
        print(length)
        #print(packet_size)
        #print(packet_size)
        data = b''
        count = 0 
        
        data_list = []
        time_init = time.time()
        try:
            while len(data_list) < length:
                print(len(data_list))
                to_read = length - len(data_list)
                count +=1028
                
                data = conn.recv(
                    4096 if to_read > 4096 else to_read)
                data_list.extend(data)
                
        finally:
            conn.close()
        
        print(time.time() - time_init)
       # print(data_list[0:5])
            
        nparr = np.array(data_list, dtype=np.uint8)
        #print(len(data_list))
        #nparr = np.fromstring(data_list, np.uint8)
        #img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        nparr = nparr.reshape((2056, 2464))
        
        cv2.imshow("img", nparr)
        cv2.waitKey(1000)
        #print(sb)
        #header = socket.recv(1)