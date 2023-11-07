# Video Semantic Segmentation with YOLO

This script processes a video file using the YOLO (You Only Look Once) object detection model to perform semantic segmentation on each frame. It allows for the application of segmentation techniques to video data, enabling the identification and localization of objects frame by frame.

## Prerequisites

Before running this script, ensure you have the following:
- Python 3.6 or later
- OpenCV library installed (`cv2`)
- `ultralytics` YOLO module installed
- NumPy library installed
- PIL (Python Imaging Library) installed
- YOLO model weights (`.pt` file)

## Installation

Clone the repository and navigate to the directory:

```bash
git clone [repository_url]
cd [repository_directory]
```

Install the required Python packages (if not already installed):

```bash
pip install opencv-python-headless numpy Pillow
```


## Usage
The script can be run from the command line by providing the required arguments. Here's how to run the script:

```
python video_segmentation.py --input_video_path "./data/input_video.mp4" \
                             --output_video_path "./data/output_video.mp4" \
                             --weights_path "./weights/best.pt" \
                             --num_frames 200 \
                             --black_height 800
```

## Arguments
- --input_video_path (required): Path to the input video file.
- --output_video_path (required): Path for saving the output video file.
- --weights_path (required): Path to the YOLO model weights file.
- --num_frames (optional): Number of frames to process from the input video. Default is 200.
- --black_height (optional): Height of the top and bottom black bars to be added to the frame. Default is 800.