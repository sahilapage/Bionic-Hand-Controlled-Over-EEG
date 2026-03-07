# Dataset Recording — OAK-D RGB + Depth

Records synchronized **RGB frames** and **depth maps** from an [OAK-D](https://docs.luxonis.com/) stereo camera using [DepthAI](https://github.com/luxonis/depthai-python).

## What it saves

```
dexmv_data/session_001/
├── rgb/          # 1080p BGR PNGs
├── depth/        # Aligned depth maps (.npy, meters)
├── intrinsics.json
└── timestamps.txt
```

## Usage

```bash
pip install depthai opencv-python numpy
python record.py        # press 'q' to stop
```

Sessions are auto-numbered (`session_001`, `session_002`, …). The first 5 frames are skipped for camera warm-up.
