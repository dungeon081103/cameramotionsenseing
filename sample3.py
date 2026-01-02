import cv2
import time

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Initialize the motion detection variables
motion_detected = False
snapshot_taken = False

while True:
    # Read the current frame from the video feed
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for motion detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # If it's the first frame, initialize the background model
    if not motion_detected:
        background_model = gray_frame
        motion_detected = True
        continue

    # Calculate the absolute difference between the current frame and the background model
    frame_delta = cv2.absdiff(background_model, gray_frame)
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if motion is detected
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion_detected = True

    # Take a snapshot if motion is detected and a snapshot hasn't been taken yet
    if motion_detected and not snapshot_taken:
        snapshot_taken = True
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.imwrite(f'motion_snapshot_{current_time}.jpg', frame)

    # Display the resulting frame
    cv2.imshow("Motion Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
