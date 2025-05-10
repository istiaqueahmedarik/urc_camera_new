import cv2
import numpy as np
import socket
import struct
import math
import sys
import threading
import time

#!/usr/bin/env python

frames = {}
selected_port = None
ports_list = []

bandwidth_lock = threading.Lock()
total_bytes_received = 0
bandwidth_mbps = 0.0


class FrameSegment:
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64

    def __init__(self, sock, dest_ip, port):
        self.sock = sock
        self.target = (dest_ip, port)

    def udp_frame(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
        jpg = cv2.imencode('.jpg', img, encode_param)[1].tobytes()
        size = len(jpg)
        parts = math.ceil(size/self.MAX_IMAGE_DGRAM)
        idx = 0
        while parts:
            chunk = jpg[idx:idx+self.MAX_IMAGE_DGRAM]
            packet = struct.pack('B', parts) + chunk
            self.sock.sendto(packet, self.target)
            idx += self.MAX_IMAGE_DGRAM
            parts -= 1


def dump_buffer(s):
    while True:
        seg, _ = s.recvfrom(2**16)
        if seg and seg[0] == 1:
            break


def receiver_thread(port):
    global frames, total_bytes_received, bandwidth_lock
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    dump_buffer(s)
    buf = b''

    while True:
        seg, _ = s.recvfrom(2**16)
        if not seg:
            break
        with bandwidth_lock:
            total_bytes_received += len(seg)
        cnt = seg[0]
        buf += seg[1:]
        if cnt == 1:
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames[port] = img
            buf = b''
    s.close()


def dashboard_mouse_callback(event, x, y, _flags, param):
    global selected_port, ports_list
    if event == cv2.EVENT_LBUTTONDOWN and x >= 900:
        thumb_h = param['height'] // len(ports_list)
        idx = y // thumb_h
        if idx < len(ports_list):
            selected_port = ports_list[idx]


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ('tx', 'rx'):
        print("Usage:\n"
              "  tx mode: multi_image_cast.py tx [camera_idx] [dest_ip] [port]\n"
              "  rx mode: multi_image_cast.py rx [port1,port2,...]\n")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == 'tx':
        cam_idx = int(sys.argv[2])
        dst_ip = sys.argv[3]
        dst_port = int(sys.argv[4])
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fs = FrameSegment(sock, dst_ip, dst_port)

        # Use v4l2 backend explicitly
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)

        # Configure for low CPU usage
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'))  # Use MJPEG
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Low resolution width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Low resolution height
        cap.set(cv2.CAP_PROP_FPS, 10)  # Set to 15fps

        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fs.udp_frame(frame)
        cap.release()
        sock.close()

    else:  # rx
        global ports_list, selected_port, bandwidth_mbps, total_bytes_received, bandwidth_lock
        ports_list = list(map(int, sys.argv[2].split(',')))
        for p in ports_list:
            t = threading.Thread(target=receiver_thread,
                                 args=(p,), daemon=True)
            t.start()
        selected_port = ports_list[0] if ports_list else None

        _W, H = 1200, 800
        cv2.namedWindow('Dashboard')
        cv2.setMouseCallback('Dashboard', dashboard_mouse_callback,
                             param={'height': H})

        # Bandwidth calculation thread
        def bandwidth_monitor():
            global total_bytes_received, bandwidth_mbps, bandwidth_lock
            prev_bytes = 0
            while True:
                time.sleep(1)
                with bandwidth_lock:
                    bytes_now = total_bytes_received
                    total_bytes_received = 0
                bandwidth_mbps = (bytes_now * 8) / \
                    1_000_000  # bits to megabits

        threading.Thread(target=bandwidth_monitor, daemon=True).start()

        while True:
            if selected_port in frames:
                focus = cv2.resize(frames[selected_port], (900, H))
            else:
                focus = np.zeros((H, 900, 3), np.uint8)

            thumbs = []
            th = H // len(ports_list) if ports_list else H
            for p in ports_list:
                img = frames.get(p, np.zeros((th, 300, 3), np.uint8))
                timg = cv2.resize(img, (300, th))
                cv2.putText(timg, f"Port {p}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                thumbs.append(timg)

            right = np.vstack(thumbs) if thumbs else np.zeros(
                (H, 300, 3), np.uint8)
            # ensure right pane matches height H
            h_right = right.shape[0]
            if h_right < H:
                pad = np.zeros((H - h_right, right.shape[1], 3), np.uint8)
                right = np.vstack((right, pad))
            elif h_right > H:
                right = right[:H]
            dash = np.hstack((focus, right))

            # Overlay bandwidth info
            cv2.putText(
                dash,
                f"Total Bandwidth: {bandwidth_mbps:.2f} Mbps",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

            cv2.imshow('Dashboard', dash)
            if cv2.waitKey(30) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

