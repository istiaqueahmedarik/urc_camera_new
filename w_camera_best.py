# pip uninstall opencv-python
# sudo apt install build-essential cmake git python3-dev python3-numpy libavcodec-dev libavformat-dev libswscale-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev libpng-dev  libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev libopencv-dev x264 libx264-dev libssl-dev ffmpeg
# python -m pip install --no-binary opencv-python opencv-python
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
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
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


def create_dashboard_layout(frames, ports_list, width=1920, height=1080):
    dashboard = np.zeros((height, width, 3), np.uint8)

    num_cameras = len(ports_list)

    if num_cameras == 0:
        return dashboard

    elif num_cameras == 1:
        if ports_list[0] in frames:
            img = frames[ports_list[0]]
            if img is not None:
                dashboard = cv2.resize(img, (width, height))
                cv2.putText(dashboard, f"Port {ports_list[0]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        if num_cameras <= 2:
            grid_rows, grid_cols = 1, 2
        elif num_cameras <= 4:
            grid_rows, grid_cols = 2, 2
        elif num_cameras <= 6:
            grid_rows, grid_cols = 2, 3
        elif num_cameras <= 9:
            grid_rows, grid_cols = 3, 3
        else:  # 10 cameras
            grid_rows, grid_cols = 3, 4

        # Calculate cell dimensions
        cell_width = width // grid_cols
        cell_height = height // grid_rows

        # Place each camera in the grid
        for i, port in enumerate(ports_list[:10]):  # Limit to 10 cameras
            if port not in frames or frames[port] is None:
                continue

            # Calculate grid position
            row = i // grid_cols
            col = i % grid_cols

            if row >= grid_rows:  # Safety check
                continue

            # Resize and place image
            img = cv2.resize(frames[port], (cell_width, cell_height))
            y_start, y_end = row * cell_height, (row + 1) * cell_height
            x_start, x_end = col * cell_width, (col + 1) * cell_width

            dashboard[y_start:y_end, x_start:x_end] = img

            # Add port label
            cv2.putText(dashboard, f"Port {port}", (x_start + 10, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add bandwidth info overlay
    cv2.putText(
        dashboard,
        f"Total Bandwidth: {bandwidth_mbps:.2f} Mbps",
        (10, height-40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    return dashboard


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
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('h', 'e', 'v', '1'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Low resolution width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Low resolution height
        cap.set(cv2.CAP_PROP_FPS, 15)  # Set to 15fps

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
        global ports_list, bandwidth_mbps, total_bytes_received, bandwidth_lock
        ports_list = list(map(int, sys.argv[2].split(',')))
        for p in ports_list:
            t = threading.Thread(target=receiver_thread,
                                 args=(p,), daemon=True)
            t.start()

        # Set fullscreen window
        cv2.namedWindow('Dashboard', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            'Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get screen resolution
        screen_width = 1920  # Default, can be made dynamic if needed
        screen_height = 1080  # Default, can be made dynamic if needed

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
            dashboard = create_dashboard_layout(
                frames, ports_list, screen_width, screen_height)
            cv2.imshow('Dashboard', dashboard)
            if cv2.waitKey(30) == 27:
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
