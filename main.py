import cv2
import numpy as np
from ultralytics import YOLO

# Đường dẫn video
cap = cv2.VideoCapture("../src/vehicle_counting.mp4")
assert cap.isOpened(), "Lỗi khi đọc file video"

# Định nghĩa các vạch đếm (line) – 2 điểm
lines = {
    "line-01": [(163, 1628), (583, 1681)],
    "line-02": [(740, 1509), (1212, 1528)],
    "line-03": [(1302, 1366), (1712, 1371)],
    "line-04": [(2299, 1404), (2718, 1380)],
    "line-05": [(2256, 990), (2485, 975)],
}

# Màu từng line
line_colors = {
    "line-01": (0, 0, 255),
    "line-02": (0, 255, 0),
    "line-03": (255, 0, 0),
    "line-04": (0, 255, 255),
    "line-05": (255, 0, 255),
}

# Khởi tạo YOLO
model = YOLO('../src/yolo11n.pt')

# Thông tin video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Theo dõi lịch sử vị trí và ID đã đếm
object_positions = {}  # {id: (x, y)}
line_crossed_ids = {line_name: set() for line_name in lines}

# Hàm kiểm tra 2 đoạn thẳng có cắt nhau không
def lines_intersect(A, B, C, D):
    def ccw(X, Y, Z):
        return (Z[1]-X[1]) * (Y[0]-X[0]) > (Y[1]-X[1]) * (Z[0]-X[0])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video đã xử lý xong.")
        break

    results = model.track(frame, persist=True, tracker='../src/botsort.yaml', device=0)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is None:
                continue
            object_id = int(box.id.item())
            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            current_pos = (x_center, y_center)

            if object_id in object_positions:
                prev_pos = object_positions[object_id]

                for line_name, (pt1, pt2) in lines.items():
                    if object_id not in line_crossed_ids[line_name]:
                        if lines_intersect(prev_pos, current_pos, pt1, pt2):
                            line_crossed_ids[line_name].add(object_id)
                            print(f"Object {object_id} crossed {line_name}")

            object_positions[object_id] = current_pos

    # Vẽ kết quả
    annotated_frame = results[0].plot()

    for line_name, (pt1, pt2) in lines.items():
        color = line_colors[line_name]
        cv2.line(annotated_frame, pt1, pt2, color, 4)
        count = len(line_crossed_ids[line_name])
        label_pos = (pt1[0], pt1[1] - 10)
        cv2.putText(annotated_frame, f"{line_name}: {count}", label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)  # Tăng font size và độ dày

    total_unique = len(set().union(*line_crossed_ids.values()))
    cv2.putText(annotated_frame, f"Total: {total_unique}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)  # Dòng tổng to nhất

    video_writer.write(annotated_frame)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
