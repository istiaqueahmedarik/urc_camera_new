import pyzed.sl as sl
import cv2
import numpy as np
import time


def main():
    # Initialize ZED camera
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {err}")
        return

    # Create image objects
    left_image = sl.Mat()

    # Create a list to store frames
    frames = []

    print("Capturing video for 10 seconds...")
    start_time = time.time()

    # Sample frames every 0.5 seconds (20 frames total over 10 seconds)
    while time.time() - start_time < 20:
        # Grab a new frame from camera
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            # Convert to OpenCV format and add to frames
            img = left_image.get_data()
            # Convert from RGBA to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            # Also retrieve depth map and position information
            depth_map = sl.Mat()
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # Enable positional tracking if not already enabled
            if frames == []:  # Only do once at the start
                tracking_params = sl.PositionalTrackingParameters()
                zed.enable_positional_tracking(tracking_params)

            # Get camera position
            camera_pose = sl.Pose()
            zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)

            # Store frame with metadata
            frames.append({
                "image": img_rgb,
                "depth": depth_map.get_data(),
                "position": {
                    "x": camera_pose.get_translation().get()[0],
                    "y": camera_pose.get_translation().get()[1],
                    "z": camera_pose.get_translation().get()[2],
                    "roll": camera_pose.get_euler_angles()[0],
                    "pitch": camera_pose.get_euler_angles()[1],
                    "yaw": camera_pose.get_euler_angles()[2]
                },
                "timestamp": time.time() - start_time
            })

            # Small delay to avoid capturing too many frames
            time.sleep(0.5)

    # Close the camera
    zed.disable_positional_tracking()
    zed.close()
    print(f"Captured {len(frames)} frames")

    # Extract just the images for stitching
    frame_images = [frame["image"] for frame in frames]

    # Create panorama using OpenCV stitcher
    print("Creating panorama...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, panorama = stitcher.stitch(frame_images)

    if status != cv2.Stitcher_OK:
        print(f"Stitching failed with error code {status}")
        return

    print("Panorama created successfully")

    # Create a depth map visualization
    depth_maps = [frame["depth"] for frame in frames]

    # Create a composite depth visualization (simplified approach)
    # In a real application, you would need to align depth maps with the panorama
    depth_composite = np.zeros_like(panorama)
    if len(depth_maps) > 0:
        # Use the middle depth map for visualization
        middle_depth = depth_maps[len(depth_maps)//2].copy()
        # Normalize for visualization
        depth_norm = cv2.normalize(middle_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(
            depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        # Resize to match a portion of the panorama
        h, w = panorama.shape[:2]
        depth_colored = cv2.resize(depth_colored, (w//3, h//3))
        # Overlay in corner
        panorama[20:20+depth_colored.shape[0], 20:20 +
                 depth_colored.shape[1]] = depth_colored

    # Calculate real-world scale based on camera positions
    positions_x = [frame["position"]["x"] for frame in frames]
    positions_z = [frame["position"]["z"] for frame in frames]
    if len(positions_x) > 1:
        # Estimate width in meters based on camera movement
        panorama_width_m = max(abs(max(positions_x) - min(positions_x)), 1.0)

        # Create a scale bar directly overlaid on the image
        # Position scale bar 1/3 from the bottom of the image
        bar_y = int(panorama.shape[0] * 2/3)

        # Make scale bar span most of the image width
        bar_start_x = int(panorama.shape[1] * 0.1)
        bar_end_x = int(panorama.shape[1] * 0.9)
        bar_length = bar_end_x - bar_start_x
        bar_height = 5  # thin line

        # Create semi-transparent overlay for the scale bar
        overlay = panorama.copy()
        cv2.rectangle(overlay,
                      (bar_start_x, bar_y),
                      (bar_end_x, bar_y + bar_height),
                      (255, 255, 255), -1)

        # Add line endpoints
        cv2.line(overlay, (bar_start_x, bar_y - 15),
                 (bar_start_x, bar_y + bar_height + 15), (255, 255, 255), 2)
        cv2.line(overlay, (bar_end_x, bar_y - 15), (bar_end_x,
                 bar_y + bar_height + 15), (255, 255, 255), 2)

        # Add distance label
        scale_text = f"{panorama_width_m:.2f}m"
        text_size = cv2.getTextSize(
            scale_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = bar_start_x + (bar_length - text_size[0]) // 2
        text_y = bar_y - 20  # Position text above the scale bar

        # Add text background for better visibility
        cv2.rectangle(overlay,
                      (text_x - 10, text_y - text_size[1] - 5),
                      (text_x + text_size[0] + 10, text_y + 5),
                      (0, 0, 0), -1)

        # Add the text
        cv2.putText(overlay, scale_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Apply the overlay with transparency
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, panorama, 1 - alpha, 0, panorama)

    else:
        # Add default scale information if can't calculate
        cv2.putText(panorama, "Scale: Unable to determine distance",
                    (int(panorama.shape[1]*0.3), int(panorama.shape[0]*0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(panorama, f"Created: {timestamp}", (panorama.shape[1]-400, panorama.shape[0]-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite("panorama_original.jpg", panorama)

    print("Panorama saved with depth and position information")


if __name__ == "__main__":
    main()
