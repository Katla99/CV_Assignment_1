import cv2
import numpy as np
import time

# Capture video 
video_capture = cv2.VideoCapture(0)
'''
Glósur úr tíma 
1. Velja tvo punkta
2. Telja fjölda inliers
    if N_inliers > MAX_inliers
        MAX_inliers = N_inliers
        i_1max = i_1
        i_2max = i_2
'''
def RANSAC(edge_points):
    MAX_inliers = 0 # Highest number of inliers
    best_line_points = None # Stores endpoints of best line (line with most inliers)

    if len(edge_points) > 1:
        for _ in range(1000):
            # Select two points at random
            idx1, idx2 = np.random.choice(len(edge_points), size=2, replace=False)

            point_1 = edge_points[idx1][::-1] # Reverse points to [x,y]
            point_2 = edge_points[idx2][::-1]
            
            #random_points = random.sample(range(len(edge_points)), 2)  # Randomly select two indices
            #point_1 = tuple(edge_points[random_points[0]])  # Extract first point
            #point_2 = tuple(edge_points[random_points[1]])  # Extract second point

            # Line: Ax + By + C = 0
            A = point_2[1] - point_1[1] 
            B = point_1[0] - point_2[0]
            C = point_2[0] * point_1[1] - point_1[0] * point_2[1]

            # Normalize line parameters
            A, B, C = A / np.sqrt(A**2 + B**2), B / np.sqrt(A**2 + B**2), C / np.sqrt(A**2 + B**2)

            # Geomatric distance calculations:
            distances = np.abs(A * edge_points[:, 1] + B * edge_points[:, 0] + C)

            # Count inliers
            inliers = distances <= 2 # Define that points with a distance 2 or less are inliers
            N_inliers = np.sum(inliers) # Count inliers

            # Update
            if N_inliers > MAX_inliers:
                MAX_inliers = N_inliers
                best_line_points = (point_1, point_2)

    return best_line_points, MAX_inliers


while True:
    start_time = time.time()
    ret, frame = video_capture.read()
    if not ret:
        break

    # Edge detection with Canny. Thesholds for detecting edges, 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_frame, 100, 150)

    # Find all pixels that are classified as edges
    edge_points = np.column_stack(np.where(edge_image != 0))
    #To speed up the RANSAC process a subset of the edge points is selected (500)
    k = max(1, len(edge_points) // 500)  # Sample every k-th point
    sampled_points = edge_points[::k]
    
    # Fit a line using RANSAC
    best_line, inliers_count = RANSAC(sampled_points)

    cv2.line(frame, best_line[0], best_line[1], (225, 0, 0), 2)
    print("inliers count:", inliers_count)
    
    # Processing time and FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)  # FPS calculation
    processing_time = end_time - start_time
    #print(f"Processing Time per Frame: {processing_time:.6f} seconds")

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Display results
    cv2.imshow("Edge Detection", edge_image)
    cv2.imshow("RANSAC Fitted Line", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
