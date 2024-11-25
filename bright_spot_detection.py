import cv2
import numpy as np
import time

# Byrja að taka upp, 0 er fyrir myndavélina í tölvunni (default camera) 1 fyrir símann
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

enable_bright_spot_detection = True
display_image = True

while(True):
    start_time = time.time() # Records the time at the start of processing a frame

    ret, frame = cap.read() # Captures a single frame from the video stream

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Bjartasti punkturinn
    if enable_bright_spot_detection:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
        cv2.circle(frame, max_loc, 10, (0, 255, 0), -1)

    # Rauðasti punkturinn
        red_channel = frame[:, :, 2]  # Extract the red channel
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red_channel)
        cv2.circle(frame, max_loc, 10, (0, 0, 255), -1)  # Mark the reddest spot with a red circle

    # Mæla FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)  # FPS calculation
    processing_time = end_time - start_time
    print(f"Processing Time per Frame: {processing_time:.6f} seconds")

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    

    if display_image:
        cv2.imshow('Video Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()