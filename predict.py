from ctypes import *
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from PIL import Image


def process_video_segmentation(input_video_path, output_video_path, weights_path, num_frames, black_height):
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_counter = 0  # Initialize frame counter

    while cap.isOpened() and frame_counter < num_frames:
        ret, ori_frame = cap.read()
        if not ret:
            break

        RGB_frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)

        # Here you can add code to blacken the top and bottom of the image if needed
        # RGB_frame = blacken_image_top_bottom(RGB_frame, black_height)

        result = model.predict(source=RGB_frame, save=False, imgsz=1024, show=False)

        for num in range(len(result[0].masks)):
            for polygon in result[0].masks[num].xy:
                polygon = np.array(polygon, dtype=np.int32)
                polygon = polygon.reshape((-1, 1, 2))
                cv2.polylines(ori_frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        out.write(ori_frame)

        frame_counter += 1  # Increment the frame counter

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video and apply YOLO semantic segmentation.")
    parser.add_argument('--input_video_path', type=str, help='Path to the input video file', required=True)
    parser.add_argument('--output_video_path', type=str, help='Path for saving the output video file', required=True)
    parser.add_argument('--weights_path', type=str, help='Path to the YOLO model weights', required=True)
    parser.add_argument('--num_frames', type=int, help='Number of frames to process', default=200)
    parser.add_argument('--black_height', type=int, help='Height of the top and bottom black bars', default=800)

    args = parser.parse_args()
    process_video_segmentation(
        input_video_path=args.input_video_path,
        output_video_path=args.output_video_path,
        weights_path=args.weights_path,
        num_frames=args.num_frames,
        black_height=args.black_height
    )
