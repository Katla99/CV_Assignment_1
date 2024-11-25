import cv2
import time
import numpy as np

# Start capturing video from the webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
while True:
    start_time = time.time()  # Start time to measure FPS

    ret, frame = cap.read()  # Capture a frame
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize variables for the brightest spot
    max_brightness = -1
    brightest_loc = None

    # Find the brightest spot using a double for-loop
    for i in range(gray.shape[0]):  # Loop through rows
        for j in range(gray.shape[1]):  # Loop through columns
            if gray[i, j] > max_brightness:
                max_brightness = gray[i, j]
                brightest_loc = (j, i)  # Note: OpenCV uses (x, y)

    # Mark the brightest spot with a green circle
    if brightest_loc:
        cv2.circle(frame, brightest_loc, 10, (0, 255, 0), -1)

    # Initialize variables for the reddest spot
    max_redness = -1
    reddest_loc = None

    # Extract the red channel
    red_channel = frame[:, :, 2]

    # Find the reddest spot using a double for-loop
    for i in range(red_channel.shape[0]):  # Loop through rows
        for j in range(red_channel.shape[1]):  # Loop through columns
            if red_channel[i, j] > max_redness:
                max_redness = red_channel[i, j]
                reddest_loc = (j, i)  # Note: OpenCV uses (x, y)

    # Mark the reddest spot with a red circle
    if reddest_loc:
        cv2.circle(frame, reddest_loc, 10, (0, 0, 255), -1)

    # Measure FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    processing_time = end_time - start_time
    print(f"Processing Time per Frame: {processing_time:.6f} seconds")
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
