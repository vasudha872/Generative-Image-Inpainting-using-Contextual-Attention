# Generative-Image-Inpainting-using-Contextual-Attention
Project Overview

This project explores image inpainting, a computer vision task where missing parts of an image are filled in to create a coherent, visually plausible result. The method leverages Generative Adversarial Networks (GANs) and Graph Neural Networks (GNNs) with a novel Contextual Attention Mechanism to enhance image reconstruction.

Key Features:

Handles complex, non-repetitive structures in images.

Uses Graph Attention Networks (GATs) to predict missing pixel values in heatmaps.

Two-stage inpainting process: Coarse estimation and refinement.

Evaluates performance with metrics like PSNR, SSIM, and Total Variation Loss.

Datasets

The project uses heatmaps containing 2D arrays of numerical values (256 x 128, float32). The dataset is split into:

Training Data: For model learning.

Validation Data: To monitor performance during training.

Methodology

Graph Representation:

Heatmaps are transformed into graph structures.

Nodes represent pixels, and edges connect neighboring pixels.

Masking:

A binary mask simulates missing regions by setting certain node features to zero.

Two-Stage Inpainting:

Coarse Estimation: Generates a rough approximation of missing regions.

Refinement: Uses contextual attention to produce a more polished result.

Training:

Loss functions: Mean Squared Error (MSE), Perceptual Loss, and Adversarial Loss.

Optimizer: Adam with appropriate learning rates.

Evaluation:

Metrics: Mean Absolute Error (MAE), PSNR, Total Variation (TV) Loss.

System Architecture

The network includes:

Dilated Convolutional Layers: For coarse estimation.

Contextual Attention Module: Focuses on relevant areas to refine predictions.

Decoder: Synthesizes the final output by aggregating coarse and refined results.

Installation and Setup

Prerequisites:

Python 3.8+

PyTorch

PyTorch Geometric

Matplotlib

Steps:

Clone the repository:

git clone https://github.com/yourusername/inpainting-project.git

Install dependencies:

pip install -r requirements.txt

Run the preprocessing pipeline:

python preprocess.py

Train the model:

python train.py

How to Use

Place your heatmaps in the data/ directory.

Run process_heatmaps.py to load and preprocess the data.

Use plot_heatmap.py to visualize any heatmap.

After training, use inference.py to apply the model to new heatmaps and view results.

Limitations

Struggles with large missing regions or complex patterns.

Generated content may lack fine details.

Sensitive to the shape and size of masked areas.

Future Work

Implement multi-scale architectures for better global and local detail.

Develop hybrid models with iterative refinement stages.

Explore cross-domain generalization for varied applications.

