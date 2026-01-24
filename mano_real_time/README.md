# Real-Time MANO Fitting with OAK-D Pro

A PyTorch-based pipeline for real-time hand reconstruction using RGB-D input from OAK-D Pro camera, featuring hand detection and MANO parameter optimization.

## Overview

This codebase enables live RGB-D hand tracking and MANO model fitting, developed for experimentation with dexterous manipulation and imitation learning applications.

## File Structure
```
mano_real_time/
├── camera.py
├── example.py
├── hand_detector.py
├── main.py
├── mano_optimizer.py
├── model.py
└── README.md
```

## Components

### `camera.py`
Handles RGB and depth stream acquisition from the OAK-D Pro camera.

### `hand_detector.py`
Performs real-time hand detection and keypoint extraction from RGB frames.

### `model.py`
Defines the MANO hand model and related utilities.

### `mano_optimizer.py`
Optimizes MANO pose, shape, and global parameters to fit observed hand data.

### `main.py`
Primary script for running real-time MANO fitting using live camera input.

### `example.py`
Contains isolated or experimental tests for debugging and validation.

## Purpose

This research sandbox was developed to:

- Evaluate feasibility of real-time MANO fitting
- Test RGB-D based hand reconstruction
- Analyze stability and accuracy of MANO parameters from live input
- Explore integration into imitation learning and dexterous manipulation pipelines

## Usage
```bash
python main.py
```

## Notes

⚠️ **Experimental Code**: This is a research prototype and not production-ready.

- Results depend heavily on camera calibration, depth quality, and lighting conditions
- Some scripts may be incomplete or intended only for testing
- Performance optimization is ongoing

## Requirements

- OAK-D Pro camera
- PyTorch
- MANO model files

## Future Work

- Integration with imitation learning pipelines
- Performance optimization for real-time constraints
- Improved robustness to varying lighting conditions