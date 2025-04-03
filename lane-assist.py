import cv2
import numpy as np
import os
from collections import deque

VIDEOS_FOLDER = "videos"
video_path = os.path.join(VIDEOS_FOLDER, "test1.mp4")

if not os.path.exists(video_path):
    print(f"❌ Error: The file '{video_path}' does not exist.")
    exit()

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Store past lane positions for smoothing
lane_memory = {
    "left": deque(maxlen=10),  # Last 10 left lanes
    "right": deque(maxlen=10)  # Last 10 right lanes
}

# Region of Interest (ROI) Mask
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2 + 50, height // 2 + 50),
        (width // 2 - 50, height // 2 + 50)
    ]], np.int32)

    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(img, mask)

# Find and average lane lines
def average_lanes(frame, lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero

        if abs(slope) < 0.5:  # Ignore almost horizontal lines
            continue

        if slope < 0:  # Left lane
            left_lines.append((x1, y1, x2, y2))
        else:  # Right lane
            right_lines.append((x1, y1, x2, y2))

    def make_line(points, lane_type):
        if len(points) == 0:
            if lane_memory[lane_type]:  # Use previous lane position if available
                return lane_memory[lane_type][-1]
            return None

        x_coords, y_coords = [], []
        for x1, y1, x2, y2 in points:
            x_coords += [x1, x2]
            y_coords += [y1, y2]

        poly = np.polyfit(x_coords, y_coords, 1)  # Fit a line
        y1, y2 = frame.shape[0], int(frame.shape[0] * 0.6)  # Define top and bottom
        x1, x2 = int((y1 - poly[1]) / poly[0]), int((y2 - poly[1]) / poly[0])

        lane_memory[lane_type].append((x1, y1, x2, y2))  # Store in history

        # Compute moving average of the last detected lanes
        avg_x1 = int(np.mean([l[0] for l in lane_memory[lane_type]]))
        avg_y1 = int(np.mean([l[1] for l in lane_memory[lane_type]]))
        avg_x2 = int(np.mean([l[2] for l in lane_memory[lane_type]]))
        avg_y2 = int(np.mean([l[3] for l in lane_memory[lane_type]]))

        return avg_x1, avg_y1, avg_x2, avg_y2

    left_lane = make_line(left_lines, "left")
    right_lane = make_line(right_lines, "right")

    return left_lane, right_lane

# Draw filled lane space
def draw_lane_area(frame, left_lane, right_lane):
    lane_overlay = np.zeros_like(frame, dtype=np.uint8)

    if left_lane and right_lane:
        points = np.array([[
            (left_lane[0], left_lane[1]),
            (left_lane[2], left_lane[3]),
            (right_lane[2], right_lane[3]),
            (right_lane[0], right_lane[1])
        ]], dtype=np.int32)

        cv2.fillPoly(lane_overlay, points, (0, 255, 0))  # Green filled area

    return cv2.addWeighted(frame, 0.8, lane_overlay, 0.3, 0)

# Draw dotted lane guide
def draw_dotted_lines(frame, lane):
    if lane:
        x1, y1, x2, y2 = lane
        for i in range(0, 10):  # Create dots along the line
            alpha = i / 10.0
            x = int(x1 * (1 - alpha) + x2 * alpha)
            y = int(y1 * (1 - alpha) + y2 * alpha)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Red dots

# Lane detection
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

    left_lane, right_lane = average_lanes(frame, lines)
    frame = draw_lane_area(frame, left_lane, right_lane)  # Green shaded region

    draw_dotted_lines(frame, left_lane)
    draw_dotted_lines(frame, right_lane)

    return frame, edges, roi


#----------------------------------------------------------------------------------------------------------------------


while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  

    processed_frame, edges, roi = detect_lanes(frame)

    cv2.imshow("Edge Detection", edges)
    cv2.imshow("Region of Interest", roi)
    cv2.imshow("Lane Detection (Shaded + Dots) [Stable]", processed_frame)  # New visualization

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
