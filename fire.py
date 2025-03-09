import cv2
import concurrent.futures
from deepface import DeepFace

# Load Haar cascade for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set up the video capture.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

reference_img = cv2.imread("referemees.jpg")

def check_face(frame):
    """
    Detects the face using Haar cascades, then performs face verification.
    Returns True if a match is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return False

    # Assume the largest detected face is the one you want.
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    face_roi = frame[y:y+h, x:x+w]

    try:
        # Set enforce_detection=False to avoid errors if face detection fails inside DeepFace.
        result = DeepFace.verify(face_roi, reference_img.copy(), enforce_detection=False)
        return result.get('verified', False)
    except Exception:
        return False

# Create a thread pool with one worker.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
future = None

while True:
    ret, frame = cap.read()
    if ret:
        # Check for face verification every 30 frames.
        if counter % 30 == 0:
            if future is None or future.done():
                future = executor.submit(check_face, frame.copy())
        
        counter += 1

        if future is not None and future.done():
            try:
                face_match = future.result()
            except Exception:
                face_match = False

        # Visual feedback.
        if face_match:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
            cv2.putText(frame, "Face Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
            cv2.putText(frame, "No Face Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

executor.shutdown(wait=True)
cap.release()
cv2.destroyAllWindows()
