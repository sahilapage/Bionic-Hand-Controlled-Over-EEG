import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def detect(self, rgb):
        h, w, _ = rgb.shape
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        lm = results.multi_hand_landmarks[0].landmark
        joints_2d = np.array([[p.x * w, p.y * h] for p in lm])

        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm])
        cv2.fillConvexPoly(mask, pts, 1)

        return joints_2d, mask