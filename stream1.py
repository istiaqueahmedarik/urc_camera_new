import cv2
import socket
import struct
import math
import sys
import threading
import time
import pyudev


DEFAULT_DEST_IP = "127.0.0.1"
STARTING_PORT = 1234
MAX_CAMERAS = 10
JPEG_QUALITY = 25
CAMERA_FPS = 15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

active_cameras = {}
next_streaming_port = STARTING_PORT
available_ports = []
cameras_lock = threading.Lock()


def _recycle_port(port_number, device_node_for_logging="Unknown device"):
    global available_ports
    if port_number != -1 and port_number not in available_ports:
        available_ports.append(port_number)
        available_ports.sort()
        print(
            f"Port {port_number} for {device_node_for_logging} recycled. Available ports: {available_ports}")
    elif port_number in available_ports:
        print(
            f"Port {port_number} for {device_node_for_logging} was already in available_ports list (call to _recycle_port).")
    elif port_number == -1:
        print(
            f"Warning: Attempted to recycle invalid port -1 for {device_node_for_logging}.")


class FrameSegment:
    MAX_DGRAM = 2**16
    MAX_IMAGE_DGRAM = MAX_DGRAM - 64

    def __init__(self, sock, dest_ip, port):
        self.sock = sock
        self.target = (dest_ip, port)

    def udp_frame(self, img):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        result, encimg = cv2.imencode('.jpg', img, encode_param)

        if not result:
            print(f"Error encoding frame for {self.target}")
            return

        jpg_bytes = encimg.tobytes()
        size = len(jpg_bytes)

        if size == 0:
            print(f"Warning: Encoded image is empty for {self.target}")
            return

        total_num_packets_for_frame = math.ceil(size / self.MAX_IMAGE_DGRAM)
        if total_num_packets_for_frame == 0 and size > 0:
            total_num_packets_for_frame = 1

        idx = 0
        for part_index in range(total_num_packets_for_frame):
            chunk = jpg_bytes[idx: idx + self.MAX_IMAGE_DGRAM]
            remaining_packets = total_num_packets_for_frame - part_index
            try:
                packet = struct.pack('B', remaining_packets) + chunk
                self.sock.sendto(packet, self.target)
            except socket.error as e:
                print(
                    f"Socket error sending to {self.target} (part {part_index+1}/{total_num_packets_for_frame}): {e}")
                return
            idx += self.MAX_IMAGE_DGRAM


def camera_stream_thread(device_node, dest_ip, stream_port, stop_event):
    print(f"Starting stream for {device_node} to {dest_ip}:{stream_port}")
    cap = None
    sock = None
    try:
        cap = cv2.VideoCapture(device_node, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(
                f"Warning: Failed to open {device_node} with V4L2 backend, trying default.")
            cap = cv2.VideoCapture(device_node)
        if not cap.isOpened():
            print(
                f"Error: Cannot open camera {device_node}. Stream thread exiting.")
            return
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # if cap.getBackendName() == 'GStreamer':
        #     fourcc = cv2.VideoWriter_fourcc('h', 'v', 'c', '1')
        # else:
        #     fourcc = cv2.VideoWriter_fourcc('h', 'v', 'c', '1')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc_str = "".join(
            [chr((actual_fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        print(f"Camera {device_node} opened. Effective settings: "
              f"{actual_width}x{actual_height} @ {actual_fps:.2f} FPS, FOURCC: {actual_fourcc_str}")
        print(
            f"  (Requested: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS)")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fs = FrameSegment(sock, dest_ip, stream_port)
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(
                    f"Error reading frame from {device_node}. Stopping stream.")
                break
            if frame is None:
                print(
                    f"Warning: Read empty frame from {device_node}. Skipping.")
                continue
            fs.udp_frame(frame)
    except Exception as e:
        print(f"Exception in camera_stream_thread for {device_node}: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        if sock:
            sock.close()
        print(f"Stopped stream for {device_node} (port {stream_port})")
        with cameras_lock:
            if device_node in active_cameras and active_cameras[device_node]['port'] == stream_port:
                print(
                    f"Thread for {device_node} (port {stream_port}) terminated unexpectedly. Cleaning up resources.")
                cam_info = active_cameras.pop(
                    device_node, None)
                if cam_info:
                    _recycle_port(cam_info['port'], device_node)


def _add_camera_device(device):
    global next_streaming_port
    global available_ports
    device_node = device.device_node
    if device_node in active_cameras:
        print(f"Camera {device_node} is already active or being processed.")
        return
    if len(active_cameras) >= MAX_CAMERAS:
        print(
            f"Max camera limit ({MAX_CAMERAS}) reached. Ignoring {device_node}.")
        return
    print(
        f"Attempting to verify and start stream for new camera: {device_node}")
    cap_test = cv2.VideoCapture(device_node, cv2.CAP_V4L2)
    if not cap_test.isOpened():
        cap_test = cv2.VideoCapture(device_node)
    if cap_test.isOpened():
        cap_test.release()
        print(f"Camera {device_node} verified as a video device.")
        port_to_use = -1
        if available_ports:
            port_to_use = available_ports.pop(0)
            print(f"Reusing freed port {port_to_use} for {device_node}")
        else:
            port_to_use = next_streaming_port
            next_streaming_port += 1
            print(f"Assigning new port {port_to_use} for {device_node}")
        stop_event = threading.Event()
        thread = threading.Thread(target=camera_stream_thread,
                                  args=(device_node, DEFAULT_DEST_IP,
                                        port_to_use, stop_event),
                                  daemon=True)
        active_cameras[device_node] = {
            'thread': thread, 'port': port_to_use, 'stop_event': stop_event}
        thread.start()
    else:
        print(
            f"Failed to open {device_node}. It might not be a usable video camera or is already in use exclusively.")


def _remove_camera_device(device_node):
    if device_node in active_cameras:
        print(f"Removing camera: {device_node}")
        cam_info = active_cameras.pop(device_node)
        cam_info['stop_event'].set()
        _recycle_port(cam_info['port'], device_node)
        print(
            f"Stream for {device_node} (port {cam_info['port']}) signaled to stop.")
    else:
        print(
            f"Camera {device_node} not found in active list for removal, or already removed.")


def handle_device_event(device):
    action = device.action
    if not device or not device.device_node or not device.device_node.startswith('/dev/video'):
        return
    device_node = device.device_node
    print(f"UDEV Event: {action} for {device_node}")
    with cameras_lock:
        if action == 'add':
            _add_camera_device(device)
        elif action == 'remove':
            _remove_camera_device(device_node)


def main():
    print("Starting camera detection and streaming script...")
    print(
        f"Streaming to IP: {DEFAULT_DEST_IP}, starting from port: {STARTING_PORT}")
    print(f"Max cameras: {MAX_CAMERAS}. JPEG Quality: {JPEG_QUALITY}.")
    print("Press Ctrl+C to exit.")
    context = pyudev.Context()
    monitor = pyudev.Monitor.from_netlink(context)
    monitor.filter_by(subsystem='video4linux')
    observer = pyudev.MonitorObserver(
        monitor, callback=handle_device_event, name='udev-monitor')
    observer.daemon = True
    observer.start()
    print("Scanning for existing cameras...")
    with cameras_lock:
        for device in context.list_devices(subsystem='video4linux'):
            if device.device_node and device.device_node.startswith('/dev/video'):
                print(f"Found existing device: {device.device_node}")
                _add_camera_device(device)
    try:
        while True:
            time.sleep(5)
            with cameras_lock:
                for dev_node, info in list(active_cameras.items()):
                    if not info['thread'].is_alive():
                        print(
                            f"Detected dead thread for {dev_node} (port {info['port']}). Cleaning up resources.")
                        cam_info = active_cameras.pop(
                            dev_node, None)
                        if cam_info:
                            _recycle_port(cam_info['port'], dev_node)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Shutting down gracefully...")
    finally:
        print("Stopping all camera streams...")
        threads_to_join = []
        with cameras_lock:
            for device_node, cam_info in active_cameras.items():
                print(
                    f"Signaling stop for stream: {device_node} on port {cam_info['port']}")
                cam_info['stop_event'].set()
                threads_to_join.append(cam_info['thread'])
            active_cameras.clear()
        for thread in threads_to_join:
            if thread.is_alive():
                print(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=5)
                if thread.is_alive():
                    print(
                        f"Warning: Thread {thread.name} did not finish in time.")
        if observer.is_alive():
            print("Udev observer is still alive (should exit as it's daemonic).")
        print("All streams processed for shutdown. Exiting.")


if __name__ == '__main__':
    main()
