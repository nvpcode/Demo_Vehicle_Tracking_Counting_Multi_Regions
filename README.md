# Project Vehicle Tracking and Counting with Multi-Region Support

This project demonstrates vehicle tracking and counting using YOLO, the `botsort` tracker, and multiple counting regions defined by lines.

## Overview

The project uses YOLO11n to detect vehicles in a video, tracks them using the `botsort` tracker, and counts vehicles as they cross predefined lines in the video frame. The counts for each line and the total unique vehicles counted are displayed on the output video.

## Demo
▶️ [Watch the demo video](https://drive.google.com/file/d/1qbhs_5NVjHHh6UFfFRurVG9UYuo0tMII/view?usp=drive_link)

## Prerequisites

Before running this project, ensure you have the following installed:

*   Python 3.7 or higher
*   `opencv-python`
*   `ultralytics`

You can install the necessary Python packages using pip:

```bash
pip install opencv-python ultralytics
```

## Usage

1.  **Prepare your video:**

    *   Ensure your video file (`vehicle_counting.mp4` in the `src` directory by default) is in the correct location.
    *   Update the video path in `src/main.py` if necessary.

2.  **Download the YOLO11n model:**

    *   The project uses the `yolo11n.pt` model by default. Ensure this model exists in the `src` directory.
    *   If you want to use a different YOLO11n model, download it and update the model path in `src/main.py`.

3.  **Run the `main.py` script:**

    ```bash
    python src/main.py
    ```

4.  **View the output:**

    *   The processed video with vehicle counts will be saved as `object_counting_output.avi` in the project directory.

## Configuration

The following parameters can be configured in `src/main.py`:

*   **Video Path:**
    *   Update the path to your video file using `cv2.VideoCapture()`.

*   **Counting Lines:**
    *   Define the counting lines and their colors in the `lines` and `line_colors` dictionaries. Each line is defined by two points `(x1, y1)` and `(x2, y2)`.

*   **YOLO11n Model:**
    *   Specify the path to your YOLO11n model file when initializing the YOLO model.

*   **Tracker:**
    *   The `botsort.yaml` file is used for tracking.  Ensure this file is correctly configured for your use case.

## Code Explanation

*   **Import Libraries:**
    *   Import the necessary libraries, including `cv2`, `numpy`, and `ultralytics`.

*   **Initialize Video Capture:**
    *   Load the video using `cv2.VideoCapture()`.

*   **Define Counting Lines:**
    *   Define the counting lines as a dictionary where each key is a line name and each value is a tuple containing the coordinates of the two endpoints of the line.

*   **Initialize YOLO11n Model:**
    *   Load the YOLO11n model using `YOLO()`.

*   **Process Video Frames:**
    *   Read each frame from the video.
    *   Use the YOLO11n model to detect and track objects in the frame.
    *   For each detected object, check if it has crossed any of the counting lines.
    *   Update the count for each line and the total count.
    *   Draw the counting lines, object bounding boxes, and counts on the frame.
    *   Write the processed frame to the output video.

*   **Release Resources:**
    *   Release the video capture and writer objects.

## Troubleshooting

*   **Issue: Video file not found.**
    *   Solution: Ensure the video file exists at the specified path and the path is correct in the script.

*   **Issue: YOLO11n model not found.**
    *   Solution: Ensure the YOLO11n model file exists at the specified path and the path is correct in the script.

*   **Issue: No objects are detected.**
    *   Solution: Verify that the YOLO11n model is suitable for detecting the objects in your video. You may need to train a custom model if the default model does not perform well.


## Contact

For any questions, please contact: nguyenphuongv07@gmail.com
