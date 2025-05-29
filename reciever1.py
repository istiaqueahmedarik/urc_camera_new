# pip uninstall opencv-python
# sudo apt install build-essential cmake git python3-dev python3-numpy libavcodec-dev libavformat-dev libswscale-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev libpng-dev  libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev libopencv-dev x264 libx264-dev libssl-dev ffmpeg
# python -m pip install --no-binary opencv-python opencv-python
import cv2
import numpy as np
import socket
import sys
import threading
import time

#!/usr/bin/env python

frames = {}
ports_list = []

bandwidth_lock = threading.Lock()
total_bytes_received = 0
bandwidth_mbps = 0.0

PORT_RANGE = range(1234, 1246)  # 1234-1245 inclusive
INACTIVITY_TIMEOUT = 5  # seconds

active_threads = {}
last_active = {}


def dump_buffer(s):
    while True:
        seg, _ = s.recvfrom(2**16)
        if seg and seg[0] == 1:
            break


def receiver_thread(port):
    global frames, total_bytes_received, bandwidth_lock, last_active
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    s.settimeout(1.0)
    buf = b''
    while True:
        try:
            seg, _ = s.recvfrom(2**16)
        except socket.timeout:
            # Check for inactivity
            if time.time() - last_active.get(port, 0) > INACTIVITY_TIMEOUT:
                break
            continue
        if not seg:
            break
        last_active[port] = time.time()
        with bandwidth_lock:
            total_bytes_received += len(seg)
        cnt = seg[0]
        buf += seg[1:]
        if cnt == 1:
            img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames[port] = img
                if port not in ports_list:
                    ports_list.append(port)
            buf = b''
    s.close()
    # Remove frame and port on exit
    if port in frames:
        del frames[port]
    if port in ports_list:
        ports_list.remove(port)
    if port in active_threads:
        del active_threads[port]
    if port in last_active:
        del last_active[port]


def create_dashboard_layout(frames, ports_list, width=1920, height=1080):
    dashboard = np.zeros((height, width, 3), np.uint8)

    num_cameras = len(ports_list)

    if num_cameras == 0:
        return dashboard

    elif num_cameras == 1:
        if ports_list[0] in frames:
            img = frames[ports_list[0]]
            if img is not None:
                # Maintain aspect ratio for single camera view
                h, w = img.shape[:2]
                aspect = w / h
                if width / height > aspect:  # Screen is wider than image
                    new_h = height
                    new_w = int(height * aspect)
                else:  # Image is wider than screen
                    new_w = width
                    new_h = int(width / aspect)

                # Resize maintaining aspect ratio
                img_resized = cv2.resize(img, (new_w, new_h))

                # Center the image on the dashboard
                y_offset = (height - new_h) // 2
                x_offset = (width - new_w) // 2

                dashboard[y_offset:y_offset+new_h,
                          x_offset:x_offset+new_w] = img_resized
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

            # Get original image
            img = frames[port]
            h, w = img.shape[:2]
            aspect = w / h

            # Calculate resize dimensions maintaining aspect ratio
            if cell_width / cell_height > aspect:  # Cell is wider than image
                new_h = cell_height
                new_w = int(cell_height * aspect)
            else:  # Image is wider than cell
                new_w = cell_width
                new_h = int(cell_width / aspect)

            # Resize image maintaining aspect ratio
            img_resized = cv2.resize(img, (new_w, new_h))

            # Calculate cell coordinates
            y_start = row * cell_height
            x_start = col * cell_width

            # Calculate offset to center the image in its cell
            y_offset = (cell_height - new_h) // 2
            x_offset = (cell_width - new_w) // 2

            # Place resized image in center of cell
            dashboard[y_start+y_offset:y_start+y_offset+new_h,
                      x_start+x_offset:x_start+x_offset+new_w] = img_resized

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
    global ports_list, bandwidth_mbps, total_bytes_received, bandwidth_lock, active_threads, last_active
    ports_list.clear()
    active_threads.clear()
    last_active.clear()

    # Set fullscreen window
    cv2.namedWindow('Dashboard', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        'Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_width = 1920
    screen_height = 1080

    def bandwidth_monitor():
        global total_bytes_received, bandwidth_mbps, bandwidth_lock
        while True:
            time.sleep(1)
            with bandwidth_lock:
                bytes_now = total_bytes_received
                total_bytes_received = 0
            bandwidth_mbps = (bytes_now * 8) / 1_000_000

    threading.Thread(target=bandwidth_monitor, daemon=True).start()

    # Start a receiver thread for each port in the range
    def thread_manager():
        while True:
            for port in PORT_RANGE:
                if port not in active_threads or not active_threads[port].is_alive():
                    last_active[port] = 0
                    t = threading.Thread(
                        target=receiver_thread, args=(port,), daemon=True)
                    active_threads[port] = t
                    t.start()
            time.sleep(1)

    threading.Thread(target=thread_manager, daemon=True).start()

    while True:
        # Remove inactive ports (handled in receiver_thread)
        dashboard = create_dashboard_layout(
            frames, ports_list, screen_width, screen_height)
        cv2.imshow('Dashboard', dashboard)
        if cv2.waitKey(30) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
