"""RGB-D capture and 3D hand-pose extraction.

Both modules pull in heavy optional dependencies (`depthai`, `torch`, `hamer`,
`mediapipe`), so nothing is imported here -- run them as scripts:

    python -m sohand.perception.record_rgbd
    python -m sohand.perception.hand_pose --img-folder frames/ --out-folder poses/
"""
