# CRC Tissue Classification using Wavelet Scattering Transform

## Overview

This MATLAB script performs colorectal cancer (CRC) tissue classification using wavelet scattering features combined with Support Vector Machine (SVM) classifiers. The refactored code emphasizes **reproducibility**, **interpretability**, and **clear documentation**.

## Key Improvements from Original Code

### 1. **Reproducibility**
- ✅ Fixed random seed (`RANDOM_SEED = 100`) consistently applied
- ✅ All random operations use the same seed
- ✅ Results saved to `.mat` file for later analysis
- ✅ Configuration parameters centralized in `CONFIG` structure

### 2. **Interpretability**
- ✅ Clear section headers (12 numbered sections)
- ✅ Comprehensive console output with progress messages
- ✅ Per-class performance metrics (Precision, Recall, F1-Score)
- ✅ Detailed summary at the end
- ✅ Multiple visualization figures

### 3. **Code Organization**
- ✅ Configuration separated from implementation
- ✅ Helper function included at the end
- ✅ Removed all commented-out code
- ✅ Consistent naming conventions

### 4. **Documentation**
- ✅ File header with description
- ✅ Each section clearly labeled
- ✅ Inline comments explaining key operations
- ✅ Function documentation for `helperScatImages_mean`

## Requirements

### MATLAB Version
- MATLAB R2019b or later recommended

### Required Toolboxes
1. **Wavelet Toolbox** - For wavelet scattering transform
2. **Statistics and Machine Learning Toolbox** - For SVM classifier
3. **Parallel Computing Toolbox** (optional) - For faster processing
4. **Image Processing Toolbox** - For image handling

### Dataset
- **Kather Texture 2016 Image Tiles** (5000 images)
- Download from: [Zenodo Repository](https:\/\/zenodo.org\/record\/53169)
- Dataset contains 8 tissue classes:
  - Tumor epithelium
  - Simple stroma
  - Complex stroma
  - Immune cells
  - Debris
  - Normal mucosal glands
  - Adipose tissue
  - Background

## Installation and Setup

### Step 1: Download Dataset
```matlab
% Download the Kather texture dataset from Zenodo
% Extract to a known location on your system
```

### Step 2: Update Data Path
Open `Final_CRC_Classification_Refactored.m` and modify line 20:

```matlab
DATA_PATH = "YOUR\/PATH\/TO\/Kather_texture_2016_image_tiles_5000";
```

### Step 3: Verify Toolbox Installation
```matlab
% Check if required toolboxes are installed
ver wavelet
ver stats
ver images
ver parallel  % Optional
```

## Running the Script

### Basic Execution
Simply run the script in MATLAB:
```matlab
Final_CRC_Classification_Refactored
```

The script will automatically:
1. Load and explore the data
2. Set up parallel computing (if available)
3. Split data into train\/test sets
4. Extract wavelet scattering features
5. Train SVM classifier
6. Perform cross-validation
7. Test on held-out data
8. Generate visualizations
9. Save results

### Expected Runtime
- **Without Parallel Computing**: ~30-60 minutes
- **With Parallel Computing**: ~15-30 minutes
(Depends on system specifications)

## Configuration Options

All configuration parameters are in the `CONFIG` structure (lines 17-34):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_ratio` | 0.8 | Proportion of data for training (80%) |
| `k_fold` | 10 | Number of folds for cross-validation |
| `image_size` | [150 150] | Input image dimensions |
| `invariance_scale` | 20 | Scattering invariance parameter |
| `quality_factors` | [1 1] | Wavelet quality factors |
| `num_rotations` | [6 6] | Number of rotation angles |
| `svm_kernel` | 'polynomial' | SVM kernel type |
| `svm_poly_order` | 3 | Polynomial kernel order |

### Modifying Parameters
To change any parameter, edit the `CONFIG` structure before running:

```matlab
CONFIG.k_fold = 5;                    % Change to 5-fold CV
CONFIG.svm_kernel = 'rbf';            % Use RBF kernel instead
CONFIG.train_ratio = 0.7;             % Use 70\/30 split
```

## Output and Results

### Console Output
The script provides detailed console output for each section:
```
Random seed set to: 100
Configuration complete.

SECTION 2: Loading and exploring data...
Total images loaded: 5000
...
Cross-validation accuracy: 96.45%
Test Set Accuracy: 95.80%
...
```

### Visualizations
Three figures are generated:

1. **Sample Images from Dataset**
   - Shows 20 random images with their class labels
   - Useful for data quality inspection

2. **Confusion Matrix**
   - Row-normalized and column-normalized
   - Shows classification performance per class
   - Helps identify which classes are confused

3. **Per-Class Performance**
   - Bar chart showing Precision, Recall, F1-Score
   - Allows comparison across tissue types

### Saved Results
Results are saved to `CRC_Classification_Results.mat` containing:

```matlab
results.config                 % All configuration parameters
results.random_seed            % Random seed used
results.trainfeatures          % Training feature matrix
results.testfeatures           % Testing feature matrix
results.classifier             % Trained SVM model
results.accuracy               % Test accuracy
results.crossval_accuracy      % Cross-validation accuracy
results.confusion_matrix       % Confusion matrix
results.precision              % Per-class precision
results.recall                 % Per-class recall
results.f1_score              % Per-class F1-score
results.*_time                % Timing information
```

### Loading Saved Results
```matlab
% Load previously saved results
load('CRC_Classification_Results.mat');

% Access results
fprintf('Test Accuracy: %.2f%%\n', results.accuracy);

% Use trained classifier for new predictions
new_predictions = predict(results.classifier, new_features);
```

## Reproducibility Guide

### Exact Reproduction
To reproduce the exact same results:

1. **Use the same random seed**: Already set to 100
2. **Use the same MATLAB version**: Document your version
3. **Use the same data path**: Ensure same image order
4. **Don't modify CONFIG**: Keep default parameters

### Verifying Reproducibility
Run the script twice and compare:
```matlab
% First run
Final_CRC_Classification_Refactored
load('CRC_Classification_Results.mat');
results1 = results;

% Second run
Final_CRC_Classification_Refactored
load('CRC_Classification_Results.mat');
results2 = results;

% Compare
isequal(results1.predicted_labels, results2.predicted_labels)
% Should return: true
```

## Performance Metrics Explained

### Accuracy
- Percentage of correctly classified samples
- Formula: (TP + TN) \/ Total samples

### Precision
- Of all samples predicted as class X, how many were actually class X?
- Formula: TP \/ (TP + FP)
- High precision = Few false positives

### Recall (Sensitivity)
- Of all actual class X samples, how many were correctly identified?
- Formula: TP \/ (TP + FN)
- High recall = Few false negatives

### F1-Score
- Harmonic mean of Precision and Recall
- Formula: 2 × (Precision × Recall) \/ (Precision + Recall)
- Balanced metric when classes are imbalanced

## Troubleshooting

### Common Issues

#### 1. "Cannot find image files"
**Solution**: Update `DATA_PATH` to correct location

#### 2. "Parallel pool failed to start"
**Solution**: Comment out parallel pool code or install Parallel Computing Toolbox

#### 3. "Out of memory"
**Solution**:
- Reduce number of images in dataset
- Close other applications
- Use a machine with more RAM

#### 4. "Toolbox not found"
**Solution**: Install required toolboxes via MATLAB Add-On Explorer

## Customization Examples

### Example 1: Change Train\/Test Split
```matlab
CONFIG.train_ratio = 0.7;  % 70% train, 30% test
```

### Example 2: Use Different SVM Kernel
```matlab
CONFIG.svm_kernel = 'rbf';  % Radial Basis Function
```

### Example 3: Modify Scattering Parameters
```matlab
CONFIG.invariance_scale = 30;
CONFIG.num_rotations = [8 8];  % More rotation angles
```

### Example 4: Disable Result Saving
```matlab
CONFIG.save_results = false;
```

## Code Structure

```
Final_CRC_Classification_Refactored.m
├── Section 1: Configuration & Reproducibility
├── Section 2: Data Loading & Exploration
├── Section 3: Parallel Computing Setup
├── Section 4: Data Splitting (Train\/Test)
├── Section 5: Wavelet Scattering Network Setup
├── Section 6: Feature Extraction
├── Section 7: SVM Classifier Training
├── Section 8: Cross-Validation
├── Section 9: Model Testing & Evaluation
├── Section 10: Results Visualization
├── Section 11: Save Results
├── Section 12: Summary
└── Helper Function: helperScatImages_mean
```

## References

1. **Kather et al. (2016)** - Multi-class texture analysis in colorectal cancer histology
2. **Wavelet Scattering Transform** - Mallat, S. (2012)
3. **MATLAB Documentation** - Wavelet Scattering for Image Classification

## Citation

If you use this code, please cite:

```
Kather, J. N., Weis, C.-A., Bianconi, F., Melcher, S. M., Schad, L. R.,
Gaiser, T., … Zöllner, F. G. (2016). Multi-class texture analysis in
colorectal cancer histology. Scientific Reports, 6, 27988.
```

## License

This code is provided for educational and research purposes.

## Contact

For questions or issues, please check:
- MATLAB documentation: https:\/\/www.mathworks.com\/help\/
- Kather dataset: https:\/\/zenodo.org\/record\/53169

---

**Last Updated**: 2025
**Version**: 2.0 (Refactored)
