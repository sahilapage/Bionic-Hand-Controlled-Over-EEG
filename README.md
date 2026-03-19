# EEG-Controlled Bionic Hand

Control the [Pollen Robotics Amazing Hand](https://www.pollen-robotics.com/) robot using **EEG brain signals** and **behavioral cloning**.

## Pipeline

1. **EEG Processing** (`eeg/`) — Read brain signals via Arduino, stream over LSL, classify activity state
2. **Hand Pose Detection** (`hand_pose/`) — Extract 3D hand poses from RGB video using HAMER + MediaPipe
3. **Retargeting** (`human_to_amazing_hand/`) — Map human hand gestures to robot motor commands in MuJoCo
5. **Data Collection** (`dataset_recording/`) — Record synchronized RGB-D video from OAK-D camera

## Quick Start

```bash
# Record training data
cd dataset_recording && python record.py

# Extract hand poses
python hand_pose/extract_tsv.py --img_folder path/to/images --out_folder poses/
 ```

## Folders

- **`eeg/`** — EEG acquisition, LSL streaming, brain state classification
- **`hand_pose/`** — HaMeR + MediaPipe for 3D hand pose & Task Space Vectors
- **`human_to_amazing_hand/`** — Pose retargeting to Amazing Hand (4-finger robot)
- **`dataset_recording/`** — OAK-D RGB-D camera sync recorder
- **`mjcf/`** — MuJoCo robot model (Amazing Hand + arm)
