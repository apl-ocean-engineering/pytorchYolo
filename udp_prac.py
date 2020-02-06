#!/usr/bin/python
import socket
import cv2
import numpy
import struct
import glob
import time

def send(sock, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(frame.shape)
    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
    data = numpy.array(imgencode)
    stringData = frame.tostring()
    length = struct.pack('I', int(len(stringData)))
    #print(int(len(stringData)))
    sock.send(length)
    sock.send( stringData );

TCP_IP = '127.0.0.1'
TCP_PORT = 50001

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

imgs1 = sorted(glob.glob('/home/mitchell/MarineSitu/triggered_image/MANTA1/*.jpg'))
imgs2 = sorted(glob.glob('/home/mitchell/MarineSitu/triggered_image/MANTA2/*.jpg'))
images = zip(imgs1, imgs2)

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
header = struct.pack('?', True)

while True:

    for fname1, fname2 in images:
        sock.send(header)
        frame1 = cv2.imread(fname1)

        frame2 = cv2.imread(fname2)
        send(sock, frame1)
        send(sock, frame2)
        detection = struct.unpack('?', sock.recv(1))
        print(detection)
        detection_list = []
        class_list = []
        if detection:
            for i in range(2):
                print('Here')
                detection_length = struct.unpack('I', sock.recv(4))[0]
                print(detection_length)
                for i in range(detection_length):
                    sq = struct.unpack('4f', sock.recv(16))
                    print(sq)
                    detection_list.append(detection_list)
                for i in range(detection_length):
                    c = struct.unpack('I', sock.recv(4))
                    print(c)
    #
sock.close()
decimg=cv2.imdecode(data,1)
cv2.imshow('CLIENT',decimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
