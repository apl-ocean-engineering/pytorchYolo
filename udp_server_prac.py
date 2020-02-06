#!/usr/bin/python
import socket
import cv2
import numpy
import struct

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = 'localhost'
TCP_PORT = 5002

s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(1)
conn, addr = s.accept()
header = recvall(conn,1)
length1 = recvall(conn,16)
stringData1 = recvall(conn, int(length1))
data1 = numpy.fromstring(stringData1, dtype='uint8')
print('here')
return_data = struct.pack('?', True)
conn.send(return_data)

length2 = recvall(conn,16)
stringData2 = recvall(conn, int(length2))
data2 = numpy.fromstring(stringData2, dtype='uint8')
#

decimg1=cv2.imdecode(data1,1)
decimg2=cv2.imdecode(data2,1)
cv2.imshow('SERVER1',decimg1)
cv2.imshow('SERVER2',decimg2)
cv2.waitKey(0)
return_data = struct.pack('?', True)
conn.send(return_data)

cv2.destroyAllWindows()
s.close()
