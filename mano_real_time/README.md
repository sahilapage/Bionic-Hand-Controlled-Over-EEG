# Real-Time MANO Fitting with OAK-D Pro
The codebase was developed to experiment with live RGB–D input, hand detection, and MANO parameter optimization using a PyTorch-based pipeline.

## File Structure
mano_real_time/
├── camera.py
├── example.py
├── hand_detector.py
├── main.py
├── mano_optimizer.py
├── model.py
└── README.md

## File Descriptions

•camera.py
  Handles RGB and depth stream acquisition from the OAK-D Pro camera.

•hand_detector.py
  Performs real-time hand detection and keypoint extraction from RGB frames.

•model.py
  Defines the MANO hand model and related utilities.

•mano_optimizer.py
  Optimizes MANO pose, shape, and global parameters to fit observed hand data.

• main.py
  Primary script for running real-time MANO fitting using live camera input.

• example.py
  Contains isolated or experimental tests for debugging and validation.

## Purpose
These scripts were used to:

• Evaluate feasibility of real-time MANO fitting
• Test RGB–D based hand reconstruction
• Analyze stability and accuracy of MANO parameters from live input

This serves as a research and experimentation sandbox, especially for future integration into imitation learning and dexterous manipulation pipelines.

## Notes

• The code is experimental and not fully optimized.
• Results depend heavily on camera calibration, depth quality, and lighting.
• Some scripts may be incomplete or intended only for testing.