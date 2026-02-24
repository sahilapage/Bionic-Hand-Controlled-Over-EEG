import time
import cv2
import numpy as np
import mediapipe as mp
import mujoco
import mujoco.viewer

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "mjcf/scene.xml"

DIST_OPEN  = 0.052
DIST_CLOSE = 0.018

OPEN_CTRL  = np.array([-0.786, +0.786])
CLOSE_CTRL = np.array([+1.57,  -1.57])

# Thumb specific control
THUMB_OPEN_CTRL  = np.array([-1.57, -1.57])
THUMB_CLOSE_CTRL = np.array([+1.57, +1.57])

# Thumb distance range (tune if needed)
THUMB_DIST_OPEN  = 0.1   # thumb far from ring
THUMB_DIST_CLOSE = 0.08   # thumb touching ring


SMOOTH_ALPHA = 0.25

# --------------------------------------------------
# MediaPipe
# --------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# --------------------------------------------------
# Generic MCP–PIP distance
# --------------------------------------------------

def get_thumb_ring_distance(lm):
    thumb_mcp = np.array([
        lm.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
        lm.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
        lm.landmark[mp_hands.HandLandmark.THUMB_MCP].z,
    ])

    ring_mcp = np.array([
        lm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
        lm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
        lm.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z,
    ])

    return np.linalg.norm(thumb_mcp - ring_mcp)


def get_finger_distance(lm, mcp_id, pip_id):
    mcp = np.array([
        lm.landmark[mcp_id].x,
        lm.landmark[mcp_id].y,
        lm.landmark[mcp_id].z,
    ])

    pip = np.array([
        lm.landmark[pip_id].x,
        lm.landmark[pip_id].y,
        lm.landmark[pip_id].z,
    ])

    return np.linalg.norm(pip - mcp)


# --------------------------------------------------
# Distance → alpha
# --------------------------------------------------
def dist_to_alpha(dist):
    alpha = (DIST_OPEN - dist) / (DIST_OPEN - DIST_CLOSE)
    return np.clip(alpha, 0.0, 1.0)

def thumb_dist_to_alpha(dist):
    alpha = (THUMB_DIST_OPEN - dist) / (THUMB_DIST_OPEN - THUMB_DIST_CLOSE)
    return np.clip(alpha, 0.0, 1.0)


def apply_thumb_control(data, ctrl_idx, dist):
    alpha = thumb_dist_to_alpha(dist)
    target = THUMB_OPEN_CTRL + alpha * (THUMB_CLOSE_CTRL - THUMB_OPEN_CTRL)

    data.ctrl[ctrl_idx]     += SMOOTH_ALPHA * (target[0] - data.ctrl[ctrl_idx])
    data.ctrl[ctrl_idx + 1] += SMOOTH_ALPHA * (target[1] - data.ctrl[ctrl_idx + 1])



# --------------------------------------------------
# Apply differential control to one finger
# --------------------------------------------------
def apply_finger_control(data, ctrl_idx, dist):
    """
    ctrl_idx = starting index (0, 2, 4)
    """
    alpha = dist_to_alpha(dist)
    target = OPEN_CTRL + alpha * (CLOSE_CTRL - OPEN_CTRL)

    data.ctrl[ctrl_idx]     += SMOOTH_ALPHA * (target[0] - data.ctrl[ctrl_idx])
    data.ctrl[ctrl_idx + 1] += SMOOTH_ALPHA * (target[1] - data.ctrl[ctrl_idx + 1])


# --------------------------------------------------
# Vision
# --------------------------------------------------
def process_frame(hands, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return frame, None

    lm = res.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # Index: 5–6
    dist_index  = get_finger_distance(
        lm,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
    )

    # Middle: 9–10
    dist_middle = get_finger_distance(
        lm,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
    )

    # Ring: 13–14
    dist_ring   = get_finger_distance(
        lm,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
    )

    thumb_ring_dist = get_thumb_ring_distance(lm)

    return frame, (dist_index, dist_middle, dist_ring, thumb_ring_dist)



    


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, dists = process_frame(hands, frame)

            if dists is not None:
                d_index, d_middle, d_ring, d_thumb = dists

                # Index → ctrl 0,1
                apply_finger_control(data, 0, d_index)

                # Middle → ctrl 2,3
                apply_finger_control(data, 2, d_middle)

                # Ring → ctrl 4,5
                apply_finger_control(data, 4, d_ring)

                    
                

                apply_finger_control(data, 0, d_index)
                apply_finger_control(data, 2, d_middle)
                apply_finger_control(data, 4, d_ring)

                # Thumb → ctrl 6,7
                apply_thumb_control(data, 6, d_thumb)


                cv2.putText(
                    frame,
                    f"I:{d_index:.3f}  M:{d_middle:.3f}  R:{d_ring:.3f}, R:{d_thumb:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            for _ in range(5):
                mujoco.mj_step(model, data)

            viewer.sync()

            cv2.imshow("Hand Teleop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(max(0.0, (1 / 60) - (time.time() - start)))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()