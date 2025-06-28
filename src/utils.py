# src/utils.py

import cv2
import csv

def draw_boxes(frame, tracks):
    """
    Draw bounding boxes and IDs on the frame.
    """
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        pid = track['id']
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID: {pid}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def save_log(log_data, file_path):
    """
    Save tracking log as CSV: frame, ID, x1, y1, x2, y2
    """
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'id', 'x1', 'y1', 'x2', 'y2'])
        for row in log_data:
            writer.writerow(row)
