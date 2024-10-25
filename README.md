**Official Code for "Task-Oriented Real-time Visual Inference for IoVT Systems: A Co-design Framework of Neural Networks and Edge Deployment"**

**Problem Statement**: As video data grows, IoVT systems face latency and bandwidth issues, and edge computing struggles with balancing high model performance and low resource usage.

**Proposed Solution**: A co-design framework that optimizes neural network architecture and deployment strategies, enhancing computational efficiency on edge devices.

**Key Features**: Dynamic model structure, Roofline-based model partitioning, and multi-objective co-optimization to balance throughput and accuracy.

**Results**: The method significantly boosts throughput and accuracy compared to baselines, with stable performance across devices of varying capacities.

**Validation**: Simulated experiments show high-accuracy, real-time detection, especially for small objects, validating the method's effectiveness in IoVT systems.

1. **Dependency Installation**: First, install the required dependencies listed in `requirements.txt` to set up the correct environment.

2. **Scripts**:

   - **`rep-construction.py`**: Logic for dynamic network structure construction. This script provides a framework for dynamic structure implementation, which can be adapted to GoogLeNet and other common multi-channel networks.
   
   - **`co-design-gs.py`**: Co-Design Optimization Logic - Grid Search Version. This script is a sample logic code to showcase the feasibility of co-optimization. Users can customize this logic (linking various functions) and deploy it on specific IoT systems for optimization. Placeholders (`*`) can be replaced with basic values to run and understand the code logic.

   - **`co-design-gd.py`**: Co-Design Optimization Logic - Gradient Descent Version. Similar to the grid search, this version demonstrates co-optimization using gradient descent for fine-tuning the parameters.

3. **CMODD Dataset Access**:
   - Link: [https://pan.baidu.com/s/18Oo0xq0fOqCh2vAULKBd9g](https://pan.baidu.com/s/18Oo0xq0fOqCh2vAULKBd9g)
   - Access Code: `5aja`
   - **Note**: For privacy reasons, please contact `wjq11346@student.ubc.ca` for access to the full dataset.

This project code is intended for verifying algorithmic correctness. The complete "co-design" code, deployment code, and deployment methods will be available after the publication of the paper. Thank you for your interest!
