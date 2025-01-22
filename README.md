# RadarNet-Lite Implementation

This repository implements **RadarNet-Lite**, a point-cloud-based object detection method for autonomous driving applications, using lidar and radar data. It reproduces the object detection functionality from the original paper, with velocity estimation to be added in future updates.

### Key Features:
- Supports the **NuScenes** dataset for training and testing.
- Processes radar and lidar data into voxel representations for object detection.
- Includes tools for data preparation, visualization, and analysis.

### Quick Start:
1. **Environment Setup**: Use Python 3.7.10, PyTorch 1.7.0, and CUDA 11.0. Install additional dependencies like `nuscenes-devkit` and `opencv`.
2. **Dataset Preparation**: Organize the NuScenes dataset and preprocess it using the provided scripts.
3. **Training**: Customize training parameters (e.g., sweeps, batch size) in `main.py` and `opts.py`.
4. **Testing**: Visualize and evaluate results using provided tools (`visualize_result.py`, `test.py`, `img2video.py`).

This project provides a basic implementation of RadarNet, with room for further extension and optimization.
