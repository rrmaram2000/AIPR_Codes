# CRC Tissue Classification using Wavelet Scattering Transform

## Overview

This repository contains MATLAB code for colorectal cancer (CRC) tissue classification using wavelet scattering features combined with Support Vector Machine (SVM) classifiers. The code is designed for reproducibility and documentation of our research observations.

## Repository Contents

### Main Scripts

1. Main_CRC_Classification.m - Main classification pipeline that performs:
   - Dataset loading and exploration
   - Train/test splitting
   - Wavelet scattering feature extraction
   - Multi-class SVM classifier training
   - Cross-validation and performance evaluation
   - Results visualization and saving

2. Scattering_Path_Analysis_FINAL.m - Wavelet scattering path analysis:
   - Documents the scattering network structure
   - Retrieves and tabulates scattering paths
   - Provides frequency/scale/rotation metadata
   - Saves path information for reference

3. Vizualize_Wavelets.m - Wavelet filter visualization:
   - Generates 2D Morlet wavelet filters
   - Displays colorized complex wavelets (phase and magnitude)
   - Shows scaling function (low-pass filter)

### Output Directories

- figures/ - Contains generated visualizations (confusion matrices, performance plots, sample images, wavelet visualizations)
- wavelet_scattering_paths/ - Contains saved scattering path metadata (CSV and MAT files)

### Generated Files

- CRC_Classification_Results.mat - Saved classification results including trained model, features, and metrics

## Requirements

### MATLAB Version
- MATLAB R2019b or later recommended

### Required Toolboxes
1. Wavelet Toolbox - For wavelet scattering transform
2. Statistics and Machine Learning Toolbox - For SVM classifier
3. Parallel Computing Toolbox (optional) - For faster processing
4. Image Processing Toolbox - For image handling

### Dataset
- Kather Texture 2016 Image Tiles (5000 histology images)
- Download from: https://zenodo.org/record/53169
- Dataset contains 8 tissue classes:
  1. Tumor epithelium
  2. Simple stroma
  3. Complex stroma
  4. Immune cells
  5. Debris
  6. Normal mucosal glands
  7. Adipose tissue
  8. Background

## Setup and Installation

### Step 1: Download Dataset
Download the Kather texture dataset from Zenodo and extract to a known location on your system.

### Step 2: Update Data Path
Open Main_CRC_Classification.m and modify line 14:

DATA_PATH = "YOUR/PATH/TO/Kather_texture_2016_image_tiles_5000";

### Step 3: Verify Toolbox Installation
Check if required toolboxes are installed in MATLAB:

ver wavelet
ver stats
ver images
ver parallel

## Running the Code

### Main Classification Pipeline
Run the main classification script:

Main_CRC_Classification

The script will:
1. Load and explore the data
2. Set up parallel computing (if available)
3. Split data into train/test sets (80/20 by default)
4. Extract wavelet scattering features
5. Train SVM classifier with polynomial kernel
6. Perform 10-fold cross-validation
7. Test on held-out data
8. Generate visualizations
9. Save results to CRC_Classification_Results.mat



### Feature Space Analysis
After running the main classification, analyze the feature space:

Feature_Space_Analysis

This generates t-SNE visualizations showing how well different tissue classes separate in the feature space.

### Scattering Path Analysis
To understand the wavelet scattering network structure:

Scattering_Path_Analysis_FINAL

This documents all scattering paths and saves detailed metadata about the wavelet filters.

### Wavelet Visualization
To visualize the Morlet wavelet filters:

Vizualize_Wavelets

This creates plots showing the 2D wavelets at different scales and orientations.

## Configuration Options

The main classification script uses a CONFIG structure with the following default parameters:

- train_ratio: 0.8 (80% training, 20% testing)
- k_fold: 10 (10-fold cross-validation)
- image_size: [150 150] (input image dimensions)
- invariance_scale: 20 (scattering invariance parameter)
- quality_factors: [1 1] (wavelet quality factors)
- num_rotations: [6 6] (number of rotation angles per scale)
- svm_kernel: 'polynomial' (SVM kernel type)
- svm_poly_order: 3 (polynomial kernel order)
- svm_kernel_scale: 'auto' (kernel scale parameter)
- svm_box_constraint: 1 (SVM regularization parameter)

To modify parameters, edit the buildConfig() function in Main_CRC_Classification.m before running.

## Output and Results

### Console Output
The script provides detailed progress messages for each processing step including:
- Total images loaded and class distribution
- Training/testing set sizes
- Feature extraction progress and dimensions
- Training time
- Cross-validation accuracy
- Test set accuracy
- Per-class performance metrics

### Visualizations
Generated figures (saved to figures/ directory):

1. Sample_Images.png - Random sample of 20 images from the dataset
2. Confusion_Matrix.png - Shows classification performance with row and column normalization
3. PerClass_Performance.png - Bar chart of precision, recall, and F1-score for each tissue class

Additional visualizations from other scripts:
- t-SNE plots showing feature space separability (from Feature_Space_Analysis.m)
- Wavelet filter visualizations (from Vizualize_Wavelets.m)

### Saved Results
CRC_Classification_Results.mat contains:
- config: All configuration parameters
- random_seed: Random seed used (100)
- trainfeatures and testfeatures: Extracted feature matrices
- normalization: Mean and standard deviation used for z-score normalization
- classifier: Trained SVM model
- predicted_labels and actual_labels: Predictions and ground truth
- accuracy: Test set accuracy
- crossval_accuracy: Cross-validation accuracy
- confusion_matrix: Confusion matrix
- precision, recall, f1_score: Per-class performance metrics
- extraction_time, training_time, cv_time, prediction_time: Timing information

To load and use saved results:

load('CRC_Classification_Results.mat');
fprintf('Test Accuracy: %.2f%%\n', results.accuracy);
new_predictions = predict(results.classifier, new_features);

## Reproducibility

The code is designed for reproducibility with the following features:

- Fixed random seed (RANDOM_SEED = 100) applied consistently
- All random operations use the same seed
- Results saved to file for later analysis
- Configuration parameters centralized in CONFIG structure
- Deterministic train/test splitting

To reproduce exact results:
1. Use the same random seed (already set to 100)
2. Use the same MATLAB version
3. Use the same dataset path
4. Keep default CONFIG parameters


## Performance Metrics

- Accuracy: Percentage of correctly classified samples
- Precision: Of all samples predicted as class X, how many were actually class X
- Recall: Of all actual class X samples, how many were correctly identified
- F1-Score: Harmonic mean of precision and recall, useful for balanced evaluation




## Acknowledgments

I would like to thank Prof. Murray Loew and Dr. Elliot Levy for their guidance and mentorship throughout this research project.
