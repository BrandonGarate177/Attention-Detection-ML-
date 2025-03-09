import threading 
import cv2 

from deepface import DeepFace

#import statements hate me 

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


#must not check for face EVERY frame. Will absolutely destroy performance.
#Instead, check every 10 frames or so.
#Also, we should only check for faces if the user is in the frame.
#If the user is not in the frame, we should not be checking for faces.



#Variables
counter = 0 
face_match = False 

reference_img = cv2.imread("reference.png")

def checlk_face(frame):
    global face_match
    try:  
        #face verification
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read()

    if ret: 
        # pass
        if counter % 30 == 0:
            try:
                threading.Thread(target=checlk_face, args=(frame.copy(),)).start()

               
            except ValueError:
                #deepface does NOT tell you when it doesnt recognize a face
                #instead it throws a "ValueError" exception
                #so we can use that to our advantage
                face_match = False

        counter += 1

        if face_match:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
            cv2.putText(frame, "Face Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
            cv2.putText(frame, "No Face Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)


    key = cv2.waitKey(1)
    # if the user presses the 'q' key, break from the loop. ----------------- MUST CHECK THIS LATER IN DEVELOPMENT
    if key == ord('q'):
        break

cv2.destroyAllWindows()
# cap.release()


