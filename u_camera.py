#!/usr/bin/env python
from __future__ import division
import cv2
import numpy as np
import socket
import struct
import math
import sys
import time
import threading

global camIdx

UDP_IP = "239.0.0.18"
global UDP_PORT

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
mreq = struct.pack("4sl", socket.inet_aton(UDP_IP), socket.INADDR_ANY)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

frames = {}
selected_port = None
ports_list = []


class FrameSegment(object):
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64

    def __init__(self, sock, port, addr=UDP_IP):
        self.s = sock
        self.port = port
        self.addr = addr

    def udp_frame(self, img):
        compress_img = cv2.imencode('.jpg', img)[1]
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        compress_img = cv2.imencode('.jpg', img, params)[1]

        dat = compress_img.tostring()
        size = len(dat)
        count = math.ceil(size / (self.MAX_IMAGE_DGRAM))
        array_pos_start = 0
        while count:
            array_pos_end = min(size, array_pos_start + self.MAX_IMAGE_DGRAM)
            segment = struct.pack("B", count) + \
                dat[array_pos_start:array_pos_end]
            self.s.sendto(segment, (self.addr, self.port))
            array_pos_start = array_pos_end
            count -= 1


def dump_buffer(s):
    MAX_DGRAM = 2**16
    while True:
        seg, addr = s.recvfrom(MAX_DGRAM)
        print(seg[0])
        if struct.unpack("B", seg[0:1])[0] == 1:
            print("finish emptying buffer")
            break


def receiver_thread(port):
    global frames
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    mreq = struct.pack("4sl", socket.inet_aton(UDP_IP), socket.INADDR_ANY)
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
    s.bind((UDP_IP, port))
    dump_buffer(s)
    MAX_DGRAM = 2**16
    dat = b''
    while True:
        try:
            seg, addr = s.recvfrom(MAX_DGRAM)
        except Exception:
            break
        if struct.unpack("B", seg[0:1])[0] > 1:
            dat += seg[1:]
        else:
            dat += seg[1:]
            img = cv2.imdecode(np.frombuffer(dat, dtype=np.uint8), 1)
            if img is not None:
                frames[port] = img
            dat = b''
    s.close()


def dashboard_mouse_callback(event, x, y, flags, param):
    global selected_port, ports_list
    dashboard_width = 1200
    if event == cv2.EVENT_LBUTTONDOWN and x >= 900:
        thumb_height = param['height'] // len(ports_list)
        index = y // thumb_height
        if index < len(ports_list):
            selected_port = ports_list[index]


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ["tx", "rx"]:
        print("Usage:")
        print("  tx mode: multiimcast.py tx [camera] [port]")
        print("  rx mode: multiimcast.py rx [port1,port2,...]")
        sys.exit(1)
    if sys.argv[1] == "tx":
        global camIdx
        camIdx = int(sys.argv[2])
        global UDP_PORT
        UDP_PORT = int(sys.argv[3])
        fs = FrameSegment(sock, UDP_PORT)
        # cap = cv2.VideoCapture(camIdx)

        cap = cv2.VideoCapture(camIdx, cv2.CAP_V4L2)

        # Configure for low CPU usage
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'))  # Use MJPEG
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Low resolution width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Low resolution height
        cap.set(cv2.CAP_PROP_FPS, 10)  # Set to 15fps

        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit(1)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fs.udp_frame(frame)
        cap.release()
        sock.close()
    elif sys.argv[1] == "rx":
        global selected_port, ports_list
        ports_list = list(map(int, sys.argv[2].split(',')))
        # Start a receiver thread for each port
        for port in ports_list:
            t = threading.Thread(target=receiver_thread,
                                 args=(port,), daemon=True)
            t.start()
        selected_port = ports_list[0] if ports_list else None

        dash_width, dash_height = 1200, 800
        cv2.namedWindow('Dashboard')
        cv2.setMouseCallback('Dashboard', dashboard_mouse_callback, param={
                             'height': dash_height})

        while True:
            if selected_port in frames:
                focus = cv2.resize(frames[selected_port], (900, dash_height))
            else:
                focus = np.zeros((dash_height, 900, 3), dtype=np.uint8)
            thumbnails = []
            thumb_h = dash_height // len(
                ports_list) if ports_list else dash_height
            for port in ports_list:
                if port in frames:
                    thumb = cv2.resize(frames[port], (300, thumb_h))
                else:
                    thumb = np.zeros((thumb_h, 300, 3), dtype=np.uint8)
                cv2.putText(thumb, f"Port {port}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                thumbnails.append(thumb)
            right_panel = np.vstack(thumbnails) if thumbnails else np.zeros(
                (dash_height, 300, 3), dtype=np.uint8)
            dashboard = np.hstack((focus, right_panel))
            cv2.imshow('Dashboard', dashboard)
            if cv2.waitKey(30) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

