import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# === Camera Calibration ===
def calibrate_camera(calibration_folder, chessboard_size=(9, 6)):
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    
    # Prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # List all image files in the folder
    image_files = [os.path.join(calibration_folder, f) for f in os.listdir(calibration_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No calibration images found in the specified folder.")
        return None, None

    # Process each calibration image
    for image_path in image_files:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
        else:
            print(f"Chessboard not detected in {image_path}")

    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    if ret:
        print("Camera calibrated successfully")
    else:
        print("Camera calibration failed")
    print(np.shape(rvecs))
    print(np.shape(tvecs))
    # extrinsic parameters

    # Step 2: Visualize reprojection
    for idx, image_path in enumerate(image_files):
        img = cv2.imread(image_path)
        if idx >= len(rvecs):  # Ensure we don't exceed the number of valid calibration images
            break

        # Project 3D points into image plane
        projected_points, _ = cv2.projectPoints(obj_points[idx], rvecs[idx], tvecs[idx], camera_matrix, dist_coeffs)
        projected_points = projected_points.squeeze()  # Flatten for easy drawing

        # Draw the original detected corners (img_points)
        for corner in img_points[idx]:
            cv2.circle(img, tuple(corner.ravel().astype(int)), 5, (0, 255, 0), -1)  # Green

        # Draw the reprojected points
        for p in projected_points:
            cv2.circle(img, tuple(p.astype(int)), 5, (0, 0, 255), -1)  # Red

        # Display the result
        cv2.imshow(f"Reprojection - {os.path.basename(image_path)}", img)
        #cv2.waitKey(0)

    cv2.destroyAllWindows()
    return camera_matrix, dist_coeffs, rvecs, tvecs


# === Distance Computation ===
def compute_distance(bbox, cls, camera_matrix, dist_coeffs, rvecs, tvecs):
    """
    Estimate distance using bounding box height and camera calibration.
    """
    (x, y, w, h) = bbox  # top left corner of the box
    u = x + w/2  # bottom middle corner
    v = y + h
    pixel_homogeneous = np.array([u, v, 1]).reshape(3, 1)
    rvecs_mean = np.mean(rvecs, axis=0).reshape(3, 1)
    tvecs_mean = np.mean(tvecs, axis=0).reshape(3, 1)

    rotation_matrix, _ = cv2.Rodrigues(rvecs_mean)
    # rt = np.concatenate((rotation_matrix, tvecs_mean), axis=1)
    # rt = np.concatenate((rt, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)

    # K = np.concatenate((camera_matrix, np.zeros((3, 1))), axis=1)
    # K = np.concatenate((K, np.zeros((1, 4))), axis=0)

    Pc = np.linalg.inv(camera_matrix).dot(pixel_homogeneous)
    # Pc = np.append(Pc, 1)
    # world_coords = np.linalg.inv(rt).dot(Pc)
    if cls == 2:
        height = 1.8
    else:
        height = 2.45
    scale = 0.8 * camera_matrix[1, 1] / h
    # scale = height * camera_matrix[1, 1] / h
    scaled_Pc = Pc * scale

    # world_coords = np.matmul(np.linalg.inv(rotation_matrix), Pc) + tvecs_mean
    world_coords = np.linalg.inv(rotation_matrix).dot(scaled_Pc - tvecs_mean)

    cam_world = - np.linalg.inv(rotation_matrix).dot(tvecs_mean)
    distance = world_coords[2] - cam_world[2]

    # old:
    # focal_length = camera_matrix[1, 1]  # Focal length from calibration
    # distance = (real_world_height * focal_length) / h # add extrinsic
    return distance[0]


# === Main Pipeline ===
def main_old(video_path, calibration_folder, output_csv):
    # Step 1: Calibrate Camera
    camera_matrix, dist_coeffs = calibrate_camera(calibration_folder)
    if camera_matrix is None or dist_coeffs is None:
        print("Camera calibration failed. Exiting.")
        return
    
    # Step 2: Initialize YOLOv8 and DeepSORT
    model = YOLO('yolov8n.pt')  # Load YOLOv8 model
    tracker = DeepSort(max_age=30, n_init=3)  # Initialize DeepSORT
    
    # Step 3: Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
    
    # Step 4: Process Video
    results_list = []
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects using YOLOv8
        detections = model(frame)[0]
        dets = []
        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if (cls == 2 or cls == 7) and conf > 0.5:  # Filter cars (YOLO class ID for cars)
                w = x2 - x1
                h = y2 - y1
                dets.append(([x1, y1, w, h], conf, cls))  # ( [left,top,w,h], confidence, detection_class )
        print('dets', dets)
        # Track objects using DeepSORT
        tracks = tracker.update_tracks(dets, frame=frame)
        print('tracks: ', tracks)
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            x1, y1, w, h = map(int, track.to_ltwh())
            
            # Compute distance
            distance = compute_distance((x1, y1, w, h), camera_matrix, dist_coeffs)
            
            # Append results
            results_list.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'bbox': (x1, y1, w, h),
                'distance': distance
            })
            
            # Draw results
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {track_id} Dist: {distance:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        frame_id += 1
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save results to CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def main(video_paths, filenames, conflictTimesVideo, calibration_folder, output_csv):
    # Step 1: Calibrate Camera
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(calibration_folder)
    if camera_matrix is None or dist_coeffs is None:
        print("Camera calibration failed. Exiting.")
        return
    
    # Step 2: Initialize YOLOv8 and DeepSORT
    model = YOLO("yolo11n.pt")  # Load YOLOv8 model
    tracker = DeepSort(max_age=30, n_init=3)  # Initialize DeepSORT

    results_list = []

    for v, vid in enumerate(filenames):
        video_path = os.path.join(video_paths, vid)
        
        # Step 3: Open Video
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error opening video {video_path}")
            return
        
        # input the times required
        times_in_seconds = conflictTimesVideo[v]

        # Frame rate (ensure this matches your video properties)
        frame_rate = int(video.get(cv2.CAP_PROP_FPS))
        segment_duration = 10  # Capture 20 seconds (10 seconds before + 10 seconds after)
        
        
        # Process each time segment
        for time_in_seconds in times_in_seconds:
            start_time = max(0, time_in_seconds - 5)  # Start 10 seconds before, ensure not negative
            end_time = time_in_seconds + 5           # End 10 seconds after

            # Calculate frame range
            start_frame = int(start_time * frame_rate)
            end_frame = int(end_time * frame_rate)
            print('frame rate', frame_rate)
            # Seek to start frame
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_id = start_frame

            # Process frames in this segment
            while frame_id <= end_frame:
                ret, frame = video.read()
                if not ret:
                    print(f"End of video reached before finishing time segment {time_in_seconds}.")
                    break

                # Perform detection, tracking, and distance computation
                detections = model.predict(frame)[0]
                print('detections', detections)
                dets = []
                for box in detections.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if (cls == 2 or cls == 7) and conf > 0.5:  # Filter cars (YOLO class ID for cars)
                        w = x2 - x1
                        h = y2 - y1
                        dets.append(([x1, y1, w, h], conf, cls))  # ( [left,top,w,h], confidence, detection_class )

                # Update tracker
                tracks = tracker.update_tracks(dets, frame=frame)

                # Record results
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    track_id = track.track_id
                    x1, y1, w, h = map(int, track.to_ltwh())
                    cls = track.det_class
                    distance = compute_distance((x1, y1, w, h), cls, camera_matrix, dist_coeffs, rvecs, tvecs)

                    # Append results to list for CSV
                    results_list.append({
                        'video': v,
                        'cls': cls,
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'bbox': (x1, y1, w, h),
                        'distance': distance,
                        'time_seconds': frame_id / frame_rate
                    })

                    # Visualize bounding boxes and tracking IDs
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {track_id} Dist: {distance:.2f}m", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show video (optional for debugging)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Increment frame ID
                frame_id += 1

        # Save results to CSV
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv, index=False)


# === Example Usage ===
if __name__ == "__main__":

    filenames = ["rgb_1_2021-05-21-095030-0000.avi",
                 "rgb_1_2021-05-21-095030-0001.avi",
                 "rgb_2_2021-05-21-122803-0000.avi",
                 "rgb_2_2021-05-21-122803-0001.avi",
                 "rgb_2_2021-05-21-122803-0002.avi",
                 "rgb_3_2021-05-21-154229-0000.avi",
                 "rgb_3_2021-05-21-154229-0001.avi"]
    
    # conflictTimesVideo = [[13*60+12, 23*60+34, 32*60+48, 41*60+20, 48*60+47, 35*60+15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                       [30*60+16, 30*60+47, 31*60+28, 37*60+37, 46*60+42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                       [18*60+35, 19*60+26, 20*60+46, 27*60+29, 28*60+28, 28*60+58, 30*60+45, 34*60+50, 35*60+2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                       [61, 240+56, 352, 364, 563, 651, 12*60+32, 38*60+15, 39*60+2, 39*60+46, 43*60+35, 44*60+19, 45*60+56, 46*60+55, 48*60+30, 49*60+14, 49*60+52, 54*60+59],
    #                       [25, 89, 110, 154, 167, 21*60+29, 11*60+19, 22*60+9, 30*60+27, 33*60+14, 0, 0, 0, 0, 0, 0, 0, 0],
    #                       [405, 7*60+10, 644, 13*60+3, 19*60+42, 30*60+1, 30*60+8, 31*60+16, 33*60+21, 34*60+32, 40*60+10, 46*60+58, 51*60+13, 52*60+53, 0, 0, 0, 0],
    #                       [17*60+47, 21*60+11, 21*60+47, 24*60+11, 25*60+12, 29*60+25, 30*60+51, 31*60+29, 33*60+23, 50*60+40, 0, 0, 0, 0, 0, 0, 0, 0]]
    conflictTimesVideo = [[13*60+12, 23*60+34, 32*60+48, 41*60+20, 48*60+47, 35*60+15],
                        [30*60+16, 30*60+47, 31*60+28, 37*60+37, 46*60+42],
                        [18*60+35, 19*60+26, 20*60+46, 27*60+29, 28*60+28, 28*60+58, 30*60+45, 34*60+50, 35*60+2],
                        [61, 240+56, 352, 364, 563, 651, 12*60+32, 38*60+15, 39*60+2, 39*60+46, 43*60+35, 44*60+19, 45*60+56, 46*60+55, 48*60+30, 49*60+14, 49*60+52, 54*60+59],
                        [25, 89, 110, 154, 167, 21*60+29, 11*60+19, 22*60+9, 30*60+27, 33*60+14],
                        [405, 7*60+10, 644, 13*60+3, 19*60+42, 30*60+1, 30*60+8, 31*60+16, 33*60+21, 34*60+32, 40*60+10, 46*60+58, 51*60+13, 52*60+53],
                        [17*60+47, 21*60+11, 21*60+47, 24*60+11, 25*60+12, 29*60+25, 30*60+51, 31*60+29, 33*60+23, 50*60+40]]

    video_paths = "raw_data/"
    calibration_folder = "raw_data/calibration_images"  # Replace with your folder path
    output_csv = "vehicle_HGVpercentage_testing.csv"

    main(video_paths, filenames, conflictTimesVideo, calibration_folder, output_csv)

