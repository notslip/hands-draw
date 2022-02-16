import cv2 as cv
import mediapipe as mp


cam = cv.VideoCapture(0)

hands = mp.solutions.hands.Hands(static_image_mode = False,
                   max_num_hands =2,
                   min_tracking_confidence=0.5,
                   min_detection_confidence=0.7)

draw = mp.solutions.drawing_utils

while True:
    _, img = cam.read()
    result = hands.process(img)
    print(result)
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                print(f"{id}-{lm}")
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # cv.circle(img, (cx, cy), 4, (255, 100, 255))
                if id == 8:
                    cv.circle(img, (cx, cy), 15, (255, 100, 255), cv.FILLED)
            draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)

    cv.imshow(" hand tracking", img)
    cv.waitKey(1)

