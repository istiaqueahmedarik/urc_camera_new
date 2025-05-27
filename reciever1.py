# pip uninstall opencv-python
# sudo apt install build-essential cmake git python3-dev python3-numpy libavcodec-dev libavformat-dev libswscale-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev libpng-dev  libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev libopencv-dev x264 libx264-dev libssl-dev ffmpeg
import cv2
import numpy as np
import socket
import sys
import threading
import time
import os
from datetime import datetime

#!/usr/bin/env python

frames = {}
ports_list = []

bandwidth_lock = threading.Lock()
total_bytes_received = 0
bandwidth_mbps = 0.0

PORT_RANGE = range(1234, 1246)  # 1234-1245 inclusive
INACTIVITY_TIMEOUT = 2  # seconds

active_threads = {}
last_active = {}

# New global variables for single feed view and click detection
single_feed_port = None
feed_rects = {}
selected_save_port = None  # Track which camera is selected for saving in dashboard view

# Image saving configuration
SAVE_DIR = "saved_images"  # Directory to save images
save_feedback_timer = 0    # Timer for showing save feedback


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


def create_dashboard_layout(frames, ports_list, feed_rects_out, width=1920, height=1080):
    dashboard = np.zeros((height, width, 3), np.uint8)
    feed_rects_out.clear()  # Clear previous rects

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
                feed_rects_out[ports_list[0]] = (0, 0, width, height)

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
            feed_rects_out[port] = (
                x_start, y_start, x_end, y_end)  # Store rect

            # Add port label
            cv2.putText(dashboard, f"Port {port}", (x_start + 10, y_start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Highlight selected camera for saving with a border
            if port == selected_save_port:
                cv2.rectangle(dashboard, (x_start, y_start), (x_end, y_end),
                              # Yellow border for selected camera
                              (0, 255, 255), 3)

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


# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global single_feed_port, feed_rects
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if single_feed_port is not None:
            # Currently showing a single feed, double click to go back to dashboard
            single_feed_port = None
        else:
            # Currently showing dashboard, check if click is on a feed
            for port, rect in feed_rects.items():
                x1, y1, x2, y2 = rect
                if x1 <= x < x2 and y1 <= y < y2:
                    single_feed_port = port
                    break


def save_image(image, port=None):
    """Save image with timestamp and optional port information"""
    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create filename based on if it's a single feed or dashboard
    if port is not None:
        filename = f"{SAVE_DIR}/port_{port}_{timestamp}.jpg"
    else:
        filename = f"{SAVE_DIR}/dashboard_{timestamp}.jpg"

    # Save the image
    cv2.imwrite(filename, image)
    return filename


def main():
    global ports_list, bandwidth_mbps, total_bytes_received, bandwidth_lock, active_threads, last_active
    global single_feed_port, feed_rects, save_feedback_timer, selected_save_port

    ports_list.clear()
    active_threads.clear()
    last_active.clear()
    single_feed_port = None
    selected_save_port = None  # Initialize the selected camera
    feed_rects.clear()

    # Set fullscreen window
    cv2.namedWindow('Dashboard', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        'Dashboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Dashboard', mouse_callback)  # Set mouse callback

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
        display_frame = None  # Will hold the current frame being displayed

        if single_feed_port is not None:
            # Display single feed
            if single_feed_port in frames and frames[single_feed_port] is not None:
                img = frames[single_feed_port]
                single_view_frame = cv2.resize(
                    img, (screen_width, screen_height))
                cv2.putText(single_view_frame, f"Port {single_feed_port}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Add bandwidth info to single view
                cv2.putText(
                    single_view_frame,
                    f"Total Bandwidth: {bandwidth_mbps:.2f} Mbps",
                    (10, screen_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

                # Show save feedback if timer is active
                if save_feedback_timer > 0:
                    cv2.putText(
                        single_view_frame,
                        "Image Saved!",
                        (screen_width // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        3
                    )
                    save_feedback_timer -= 1

                display_frame = single_view_frame
                cv2.imshow('Dashboard', single_view_frame)
            else:
                # Frame for single_feed_port is not available, go back to dashboard
                # Optionally, display a "No Signal" message before switching back
                blank_screen = np.zeros(
                    (screen_height, screen_width, 3), np.uint8)
                cv2.putText(blank_screen, f"Feed for Port {single_feed_port} lost. Returning to dashboard.",
                            (50, screen_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Dashboard', blank_screen)
                cv2.waitKey(500)  # Show message briefly
                single_feed_port = None
        else:
            # Display dashboard
            dashboard = create_dashboard_layout(
                frames, ports_list, feed_rects, screen_width, screen_height)

            # Add help text for keyboard controls in dashboard view
            if len(ports_list) > 0:
                cv2.putText(
                    dashboard,
                    "Arrow keys: Select camera | S: Save selected or all | A: Save all",
                    (10, screen_height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            # Show save feedback if timer is active
            if save_feedback_timer > 0:
                cv2.putText(
                    dashboard,
                    "Image Saved!",
                    (screen_width // 2 - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 255),
                    3
                )
                save_feedback_timer -= 1

            display_frame = dashboard
            cv2.imshow('Dashboard', dashboard)

        key = cv2.waitKey(30) & 0xFF  # Use 0xFF mask for 64-bit systems

        # Handle key presses
        if key == 27:  # Esc key
            if single_feed_port is not None:
                single_feed_port = None  # Exit single view, return to dashboard
            else:
                break  # Exit application from dashboard view

        # Camera selection in dashboard mode using arrow keys
        elif key in [ord('a'), ord('A')] and single_feed_port is None:
            # 'A' key: Always save entire dashboard
            if display_frame is not None:
                save_image(display_frame)
                save_feedback_timer = 30

        # Save image when 's' key is pressed
        elif key == ord('s'):
            if display_frame is not None:
                if single_feed_port is not None:
                    # Save the current single feed
                    save_image(display_frame, single_feed_port)
                else:
                    # In dashboard view
                    if selected_save_port is not None and selected_save_port in frames:
                        # Save only the selected camera
                        img = frames[selected_save_port]
                        if img is not None:
                            save_image(img, selected_save_port)
                    else:
                        # No camera selected, save the entire dashboard
                        save_image(display_frame)

                # Show save feedback for next 30 frames (roughly 1 second)
                save_feedback_timer = 30

        # Arrow keys for camera selection in dashboard view
        elif single_feed_port is None and ports_list:
            if key == 82 or key == ord('w'):  # Up arrow or 'w'
                select_camera(True, -1)
            elif key == 84 or key == ord('s'):  # Down arrow or 's'
                select_camera(True, 1)
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                select_camera(False, -1)
            elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                select_camera(False, 1)

    cv2.destroyAllWindows()


def select_camera(is_vertical, direction):
    """Helper function to select camera in grid with arrow keys"""
    global selected_save_port, ports_list, feed_rects

    if not ports_list:
        return

    # Initialize selection if nothing is selected
    if selected_save_port is None:
        selected_save_port = ports_list[0]
        return

    # Find current selection in the grid
    if selected_save_port not in ports_list:
        selected_save_port = ports_list[0]
        return

    current_index = ports_list.index(selected_save_port)

    # Calculate grid dimensions (must match create_dashboard_layout)
    num_cameras = len(ports_list)
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

    # Find current position in grid
    current_row = current_index // grid_cols
    current_col = current_index % grid_cols

    # Calculate new position
    if is_vertical:
        current_row = (current_row + direction) % grid_rows
    else:
        current_col = (current_col + direction) % grid_cols

    new_index = current_row * grid_cols + current_col

    # Make sure it's within bounds
    if new_index < len(ports_list):
        selected_save_port = ports_list[new_index]


if __name__ == '__main__':
    main()
