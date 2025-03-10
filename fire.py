import cv2
import mediapipe as mp
import os
import time

# need to ensure that we dont use the gpu as that will ruin the preformance
# remember we want to run this on the raspberry pi
# or something like that where we dont have a gpu only a really trashy cpu


# Where to save the training data. 
attentive_dir = 'dataset/test/attentive'
distracted_dir = 'dataset/test/distracted'
os.makedirs(attentive_dir, exist_ok=True)
os.makedirs(distracted_dir, exist_ok=True)

#Face mesh model 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Video capture, the zero index refers to the default camera.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# frame width and height
ret, frame = cap.read()
if not ret:
    raise Exception("Failed to capture frame.")
height, width = frame.shape[:2]

# Define the attention zone (central region).
zone_margin = 0.25  # 25% margin on each side.
x_min = int(width * zone_margin)
x_max = int(width * (1 - zone_margin))
y_min = int(height * zone_margin)
y_max = int(height * (1 - zone_margin))

#Time based triggers 
no_face_frame_threshold = 30  # Frames without detecting a face.
no_face_counter = 0
save_cooldown = 2.0  # Seconds between automatic saves.
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # rgb color ts 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    attention_status = "attentive"  
    face_bbox = None

    if results.multi_face_landmarks:
        no_face_counter = 0
        face_landmarks = results.multi_face_landmarks[0]

        # Compute face bounding box based on landmarks.
        x_coords = [landmark.x for landmark in face_landmarks.landmark]
        y_coords = [landmark.y for landmark in face_landmarks.landmark]
        face_x_min = int(min(x_coords) * width)
        face_x_max = int(max(x_coords) * width)
        face_y_min = int(min(y_coords) * height)
        face_y_max = int(max(y_coords) * height)
        face_bbox = (face_x_min, face_y_min, face_x_max, face_y_max)
        face_detected = True

        # Draw the face bounding box.
        cv2.rectangle(frame, (face_x_min, face_y_min), (face_x_max, face_y_max), (0, 255, 0), 2)

        # Use a landmark for attention zone check (using landmark 1 as an approximation for the nose tip).
        nose_landmark = face_landmarks.landmark[1]
        nose_tip = (int(nose_landmark.x * width), int(nose_landmark.y * height))
        cv2.circle(frame, nose_tip, 3, (0, 255, 0), -1)

        # Draw the attention zone rectangle.
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # Determine attention status based on whether the nose tip is within the attention zone.
        if not (x_min <= nose_tip[0] <= x_max and y_min <= nose_tip[1] <= y_max):
            attention_status = "distracted"
    else:
        no_face_counter += 1
        if no_face_counter > no_face_frame_threshold:
            attention_status = "distracted"

    # Display attention status on the frame.
    cv2.putText(frame, attention_status, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Frame", frame)

    # Check for key presses.
    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()
    manual_save = False
    manual_label = None
    if key == ord('q'):
        break
    elif key == ord('a'):
        manual_save = True
        manual_label = "attentive"
    elif key == ord('d'):
        manual_save = True
        manual_label = "distracted"

    # Define a helper function to save cropped face images.
    def save_cropped(label):
        if face_detected and face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            # Clamp bounding box coordinates within frame boundaries.
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width)
            y2 = min(y2, height)
            face_crop = frame[y1:y2, x1:x2]
            filename = os.path.join(
                attentive_dir if label == "attentive" else distracted_dir,
                f"{label}_{int(current_time*1000)}.png"
            )
            cv2.imwrite(filename, face_crop)
            print(f"Saved {label} sample (cropped face):", filename)
        else:
            print("No face detected for cropping. Saving full frame instead.")
            filename = os.path.join(
                attentive_dir if label == "attentive" else distracted_dir,
                f"{label}_{int(current_time*1000)}.png"
            )
            cv2.imwrite(filename, frame)
            print(f"Saved full frame as {label} sample:", filename)

    # Automatic saving if cooldown time has passed.
    if current_time - last_save_time > save_cooldown:
        save_cropped(attention_status)
        last_save_time = current_time

    # Manual saving triggered by key press.
    if manual_save:
        save_cropped(manual_label)
        last_save_time = current_time

cap.release()
cv2.destroyAllWindows()
 