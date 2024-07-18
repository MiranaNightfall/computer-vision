import cv2
import time
import mediapipe as mp

# initialize capture the frame from camera (0 -> default cam index)
capture = cv2.VideoCapture(0)

# initialize mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# initialize time for fps
previousTime = 0
currentTime = 0

# initialize frame
while True:
    success, frame = capture.read()
    if not success:
        break

    frame = cv2.flip(frame, 1) # flip the frame

    # convert frame to RGB (because mediapipe only can process in RGB format)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    # draw the landmark on the hand that appeared
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for index, landmark in enumerate(handLms.landmark):
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                print(index, cx, cy)
                
                if index in [0, 4, 8, 12, 16, 20]:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 200), cv2.FILLED)
                # print(index, landmark)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    # fps configuration
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # set fps text on frame
    fpsText = "FPS: " + str(int(fps))
    position = (10, 70)
    fontFps = cv2.FONT_HERSHEY_PLAIN
    size = 3
    rgbcolor = (50, 0, 50)
    thickness = 3
    cv2.putText(frame, fpsText, position, fontFps, size, rgbcolor, thickness)

    # show the frame
    cv2.imshow('frame', frame)

    # press 'q' if want to quit the cam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release
capture.release()
cv2.destroyAllWindows()

def main():
    print()

if __name__ == "__main__":
    main()