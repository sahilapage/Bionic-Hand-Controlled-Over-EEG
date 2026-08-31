"""so-hand — dexterous manipulation stack for the Pollen Robotics Amazing Hand.

Four independent components share one robot model and one set of conventions:

    sohand.envs        MuJoCo environments (continuous in-hand cube rotation)
    sohand.rl          SAC training, evaluation and policy replay
    sohand.retarget    human hand pose -> 8 motor commands
    sohand.perception  RGB-D capture and 3D hand-pose extraction
    sohand.eeg         BioAmp EXG acquisition, LSL streaming, band classification

Nothing here imports across component boundaries, so a missing optional
dependency in one (torch, depthai, PyQt5) never breaks another.
"""

__version__ = "0.2.0"
