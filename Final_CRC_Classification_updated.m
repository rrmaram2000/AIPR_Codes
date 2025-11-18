%% ========================================================================
%  CRC TISSUE CLASSIFICATION USING WAVELET SCATTERING TRANSFORM
%  ========================================================================
%  Description: This script performs colorectal cancer (CRC) tissue
%  classification using wavelet scattering features and SVM classifier.
%
%  Dataset: Kather texture 2016 image tiles (5000 images)
%  Method: Wavelet Scattering Transform + Multi-class SVM
%
%  Author: Ritish Raghav Maram
%  Date: October 2025
%  ========================================================================

clear all; close all; clc;

%% ========================================================================
%  SECTION 1: CONFIGURATION AND REPRODUCIBILITY SETUP
%  ========================================================================

% Set random seed for reproducibility
RANDOM_SEED = 100;
rng(RANDOM_SEED);
fprintf('Random seed set to: %d\n', RANDOM_SEED);

% Configure data path (UPDATE THIS PATH TO YOUR DATA LOCATION)
DATA_PATH = "/Users/ritishmaram/Desktop/Kather_texture_2016_image_tiles_5000";

% Configuration parameters
CONFIG = struct();
CONFIG.train_ratio = 0.8;              % 80% training, 20% testing
CONFIG.num_visualization_images = 20;  % Number of images to visualize
CONFIG.k_fold = 10;                    % K-fold cross-validation
CONFIG.image_size = [150 150];         % Image dimensions
CONFIG.invariance_scale = 20;          % Wavelet scattering parameter
CONFIG.quality_factors = [1 1];        % Quality factors for scattering
CONFIG.num_rotations = [6 6];          % Number of rotations

% SVM Configuration
CONFIG.svm_kernel = 'polynomial';      % Kernel function
CONFIG.svm_poly_order = 3;             % Polynomial order
CONFIG.svm_kernel_scale = 'auto';      % Kernel scale
CONFIG.svm_box_constraint = 1;         % Box constraint

% Output configuration
CONFIG.save_results = true;            % Save intermediate results
CONFIG.output_file = 'CRC_Classification_Results.mat';
CONFIG.save_figures = true;            % Save generated figures
CONFIG.figures_dir = fullfile(pwd, 'figures');
CONFIG.figure_format = 'png';          % Default figure format

fprintf('Configuration complete.\n\n');

if CONFIG.save_figures && exist(CONFIG.figures_dir, 'dir') ~= 7
    mkdir(CONFIG.figures_dir);
    fprintf('Created figure directory: %s\n', CONFIG.figures_dir);
elseif CONFIG.save_figures
    fprintf('Figure directory already exists: %s\n', CONFIG.figures_dir);
end

%% ========================================================================
%  SECTION 2: DATA LOADING AND EXPLORATION
%  ========================================================================

fprintf('SECTION 2: Loading and exploring data...\n');

% Load image datastore
Imds = imageDatastore(DATA_PATH, ...
    'IncludeSubFolders', true, ...
    'FileExtensions', '.tif', ...
    'LabelSource', 'foldernames');

fprintf('Total images loaded: %d\n', numel(Imds.Files));
fprintf('Class labels: \n');
disp(unique(Imds.Labels));

% Visualize sample images
fig_samples = figure('Name', 'Sample Images from Dataset', 'Position', [100 100 1200 800]);
numImages = min(numel(Imds.Files), 5000);
perm = randperm(numImages, CONFIG.num_visualization_images);

for np = 1:CONFIG.num_visualization_images
    subplot(4, 5, np);
    im = imread(Imds.Files{perm(np)});
    imshow(im);
    tit = char(Imds.Labels(perm(np)));
    title(tit(4:end), 'Interpreter', 'none');
end
sgtitle('Random Sample of CRC Tissue Images');
saveFigure(fig_samples, 'Sample_Images.png', CONFIG);

fprintf('Data exploration complete.\n\n');

%% ========================================================================
%  SECTION 3: PARALLEL COMPUTING SETUP
%  ========================================================================

fprintf('SECTION 3: Setting up parallel computing...\n');

% Start parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool;
    fprintf('Parallel pool started.\n');
else
    fprintf('Parallel pool already running.\n');
end

fprintf('\n');

%% ========================================================================
%  SECTION 4: DATA SPLITTING (TRAIN/TEST)
%  ========================================================================

fprintf('SECTION 4: Splitting data into train and test sets...\n');

% Reset random seed for reproducible split
rng(RANDOM_SEED);

% Reload and shuffle the datastore
Imds = imageDatastore(DATA_PATH, ...
    'IncludeSubFolders', true, ...
    'FileExtensions', '.tif', ...
    'LabelSource', 'foldernames');
Imds = shuffle(Imds);

% Split into training and testing sets
[trainImds, testImds] = splitEachLabel(Imds, CONFIG.train_ratio);

% Display split statistics
fprintf('\nTraining set distribution:\n');
disp(countEachLabel(trainImds));

fprintf('Testing set distribution:\n');
disp(countEachLabel(testImds));

fprintf('Data splitting complete.\n\n');

%% ========================================================================
%  SECTION 5: WAVELET SCATTERING NETWORK SETUP
%  ========================================================================

fprintf('SECTION 5: Creating wavelet scattering network...\n');

% Create scattering network
sn = waveletScattering2('ImageSize', CONFIG.image_size, ...
    'InvarianceScale', CONFIG.invariance_scale, ...
    'QualityFactors', CONFIG.quality_factors, ...
    'NumRotations', CONFIG.num_rotations);

fprintf('Scattering network created with:\n');
fprintf('  - Image Size: [%d, %d]\n', CONFIG.image_size);
fprintf('  - Invariance Scale: %d\n', CONFIG.invariance_scale);
fprintf('  - Quality Factors: [%d, %d]\n', CONFIG.quality_factors);
fprintf('  - Number of Rotations: [%d, %d]\n', CONFIG.num_rotations);
fprintf('  - Total scattering paths: %d\n', numel(sn.paths));

fprintf('\n');

%% ========================================================================
%  SECTION 6: FEATURE EXTRACTION
%  ========================================================================

fprintf('SECTION 6: Extracting wavelet scattering features...\n');
fprintf('This may take several minutes...\n');

% Create tall arrays for parallel processing
Ttrain = tall(trainImds);
Ttest = tall(testImds);

% Extract features using helper function
tic;
trainfeatures_cell = cellfun(@(x) helperScatImages_mean(sn, x), Ttrain, 'Uni', 0);
testfeatures_cell = cellfun(@(x) helperScatImages_mean(sn, x), Ttest, 'Uni', 0);

% Gather results from tall arrays
fprintf('Gathering training features...\n');
Trainf = gather(trainfeatures_cell);
trainfeatures = cat(1, Trainf{:});

fprintf('Gathering testing features...\n');
Testf = gather(testfeatures_cell);
testfeatures = cat(1, Testf{:});

extraction_time = toc;

fprintf('Feature extraction complete in %.2f seconds.\n', extraction_time);
fprintf('Training feature matrix size: [%d, %d]\n', size(trainfeatures));
fprintf('Testing feature matrix size: [%d, %d]\n', size(testfeatures));
fprintf('\n');

%% ========================================================================
%  SECTION 7: SVM CLASSIFIER TRAINING
%  ========================================================================

fprintf('SECTION 7: Training SVM classifier...\n');

% Setup parallel computing options
if license('test', 'Distrib_Computing_Toolbox')
    if isempty(gcp('nocreate'))
        parpool;
    end
end

% Define SVM template
t = templateSVM('KernelFunction', CONFIG.svm_kernel, ...
    'PolynomialOrder', CONFIG.svm_poly_order, ...
    'KernelScale', CONFIG.svm_kernel_scale, ...
    'BoxConstraint', CONFIG.svm_box_constraint, ...
    'Standardize', true);

% Train multiclass SVM classifier
tic;
classifier = fitcecoc(trainfeatures, trainImds.Labels, ...
    'Learners', t, ...
    'Coding', 'onevsall', ...
    'Options', statset('UseParallel', true));
training_time = toc;

fprintf('Classifier training complete in %.2f seconds.\n', training_time);
fprintf('\n');

%% ========================================================================
%  SECTION 8: CROSS-VALIDATION
%  ========================================================================

fprintf('SECTION 8: Performing %d-fold cross-validation...\n', CONFIG.k_fold);

tic;
CVMdl = crossval(classifier, 'KFold', CONFIG.k_fold);
cvLoss = kfoldLoss(CVMdl) * 100;
crossvalAccuracy = 100 - cvLoss;
cv_time = toc;

% Format time output
if cv_time > 60
    cv_time_str = sprintf('%.2f minutes', cv_time / 60);
else
    cv_time_str = sprintf('%.2f seconds', cv_time);
end

fprintf('Cross-validation complete in %s.\n', cv_time_str);
fprintf('%d-fold Cross-validation accuracy: %.2f%%\n', CONFIG.k_fold, crossvalAccuracy);
fprintf('%d-fold Cross-validation loss: %.2f%%\n', CONFIG.k_fold, cvLoss);
fprintf('\n');

%% ========================================================================
%  SECTION 9: MODEL TESTING AND EVALUATION
%  ========================================================================

fprintf('SECTION 9: Testing classifier on held-out test set...\n');

% Make predictions
tic;
predicted_labels = predict(classifier, testfeatures);
prediction_time = toc;

% Calculate accuracy
accuracy = sum(predicted_labels == testImds.Labels) / length(testImds.Labels) * 100;

fprintf('Prediction complete in %.2f seconds.\n', prediction_time);
fprintf('Test Set Accuracy: %.2f%%\n', accuracy);
fprintf('Number of correct predictions: %d/%d\n', ...
    sum(predicted_labels == testImds.Labels), length(testImds.Labels));
fprintf('\n');

%% ========================================================================
%  SECTION 10: RESULTS VISUALIZATION
%  ========================================================================

fprintf('SECTION 10: Generating result visualizations...\n');

% Get unique classes and clean labels (remove first 3 characters: "01_")
classes = unique(testImds.Labels);
clean_labels = cellfun(@(x) x(4:end), cellstr(char(classes)), 'UniformOutput', false);

% Clean the test and predicted labels for display
testLabels_clean = categorical(cellfun(@(x) x(4:end), cellstr(char(testImds.Labels)), 'UniformOutput', false));
predictedLabels_clean = categorical(cellfun(@(x) x(4:end), cellstr(char(predicted_labels)), 'UniformOutput', false));

% Confusion matrix with brighter appearance
fig_cm = figure('Name', 'Confusion Matrix', 'Position', [100 100 900 800]);
cchart = confusionchart(testLabels_clean, predictedLabels_clean);
cchart.Title = 'Confusion Matrix - SVM Classification Results';
cchart.FontSize = 12;
cchart.RowSummary = 'row-normalized';
cchart.ColumnSummary = 'column-normalized';

% Adjust colormap for better brightness/contrast
colormap(cchart, 'turbo');  % Use brighter colormap

% Make the font bold for better visibility
cchart.FontName = 'Arial';
cchart.FontWeight = 'bold';

% Save confusion matrix figure
saveFigure(fig_cm, 'Confusion_Matrix.png', CONFIG);

% Calculate per-class metrics
num_classes = numel(classes);
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);

confMat = confusionmat(testImds.Labels, predicted_labels);

fprintf('\nPer-Class Performance Metrics:\n');
fprintf('%-20s %10s %10s %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 55));

for i = 1:num_classes
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;

    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));

    class_name = char(classes(i));
    fprintf('%-20s %9.2f%% %9.2f%% %9.2f%%\n', ...
        class_name(4:end), precision(i)*100, recall(i)*100, f1_score(i)*100);
end

% Overall metrics
avg_precision = mean(precision) * 100;
avg_recall = mean(recall) * 100;
avg_f1 = mean(f1_score) * 100;

fprintf('\n');
fprintf('Average Metrics:\n');
fprintf('  Precision: %.2f%%\n', avg_precision);
fprintf('  Recall: %.2f%%\n', avg_recall);
fprintf('  F1-Score: %.2f%%\n', avg_f1);
fprintf('\n');

% Plot per-class performance
fig_perf = figure('Name', 'Per-Class Performance', 'Position', [100 100 1000 600]);
bar_data = [precision * 100, recall * 100, f1_score * 100];
b = bar(bar_data);
set(gca, 'XTickLabel', clean_labels);
set(gca, 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Tissue Class', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Performance (%)', 'FontSize', 12, 'FontWeight', 'bold');
legend('Precision', 'Recall', 'F1-Score', 'Location', 'best', 'FontSize', 11);
title('Per-Class Classification Performance', 'FontSize', 14, 'FontWeight', 'bold');
grid on;

% Make bars brighter with better colors
b(1).FaceColor = [0.2 0.6 0.8];  % Bright blue
b(2).FaceColor = [0.8 0.4 0.2];  % Bright orange
b(3).FaceColor = [0.2 0.7 0.3];  % Bright green

% Save per-class performance figure
saveFigure(fig_perf, 'Per_Class_Performance.png', CONFIG);

fprintf('Visualizations complete.\n\n');

%% ========================================================================
%  SECTION 11: SAVE RESULTS
%  ========================================================================

if CONFIG.save_results
    fprintf('SECTION 11: Saving results to file...\n');

    % Prepare results structure
    results = struct();
    results.config = CONFIG;
    results.random_seed = RANDOM_SEED;
    results.trainfeatures = trainfeatures;
    results.testfeatures = testfeatures;
    results.trainImds = trainImds;
    results.testImds = testImds;
    results.classifier = classifier;
    results.predicted_labels = predicted_labels;
    results.actual_labels = testImds.Labels;
    results.accuracy = accuracy;
    results.crossval_accuracy = crossvalAccuracy;
    results.confusion_matrix = confMat;
    results.precision = precision;
    results.recall = recall;
    results.f1_score = f1_score;
    results.extraction_time = extraction_time;
    results.training_time = training_time;
    results.cv_time = cv_time;
    results.prediction_time = prediction_time;

    % Save to file
    save(CONFIG.output_file, 'results', '-v7.3');
    fprintf('Results saved to: %s\n', CONFIG.output_file);
end

fprintf('\n');

%% ========================================================================
%  SECTION 12: SUMMARY
%  ========================================================================

fprintf('========================================================================\n');
fprintf('CLASSIFICATION SUMMARY\n');
fprintf('========================================================================\n');
fprintf('Dataset: Kather CRC Tissue Images\n');
fprintf('Total images: %d (Train: %d, Test: %d)\n', ...
    numel(Imds.Files), numel(trainImds.Files), numel(testImds.Files));
fprintf('Number of classes: %d\n', num_classes);
fprintf('Feature dimension: %d\n', size(trainfeatures, 2));
fprintf('\nPerformance:\n');
fprintf('  %d-fold Cross-validation accuracy: %.2f%%\n', CONFIG.k_fold, crossvalAccuracy);
fprintf('  Test set accuracy: %.2f%%\n', accuracy);
fprintf('  Average precision: %.2f%%\n', avg_precision);
fprintf('  Average recall: %.2f%%\n', avg_recall);
fprintf('  Average F1-score: %.2f%%\n', avg_f1);
fprintf('\nComputation time:\n');
if extraction_time > 60
    fprintf('  Feature extraction: %.2f minutes\n', extraction_time / 60);
else
    fprintf('  Feature extraction: %.2f seconds\n', extraction_time);
end
if training_time > 60
    fprintf('  Training: %.2f minutes\n', training_time / 60);
else
    fprintf('  Training: %.2f seconds\n', training_time);
end
if cv_time > 60
    fprintf('  Cross-validation: %.2f minutes\n', cv_time / 60);
else
    fprintf('  Cross-validation: %.2f seconds\n', cv_time);
end
if prediction_time > 60
    fprintf('  Prediction: %.2f minutes\n', prediction_time / 60);
else
    fprintf('  Prediction: %.2f seconds\n', prediction_time);
end
fprintf('========================================================================\n');
fprintf('Analysis complete!\n');

%% ========================================================================
%  HELPER FUNCTION: saveFigure
%  ========================================================================
%  Saves a figure handle to disk based on configuration settings.
%  ========================================================================
function saveFigure(figHandle, filename, config)
    if ~isfield(config, 'save_figures') || ~config.save_figures
        return;
    end

    if ~isfield(config, 'figures_dir') || isempty(config.figures_dir)
        figureDir = pwd;
    else
        figureDir = config.figures_dir;
    end

    if exist(figureDir, 'dir') ~= 7
        mkdir(figureDir);
    end

    [~, ~, ext] = fileparts(filename);
    if isempty(ext)
        if isfield(config, 'figure_format') && ~isempty(config.figure_format)
            ext = ['.', config.figure_format];
        else
            ext = '.png';
        end
        filename = [filename, ext];
    end

    filePath = fullfile(figureDir, filename);
    saveas(figHandle, filePath);
    fprintf('Figure saved as: %s\n', filePath);
end

%% ========================================================================
%  HELPER FUNCTION: helperScatImages_mean
%  ========================================================================
%  This function computes the mean of wavelet scattering features
%  across spatial and channel dimensions.
%
%  Inputs:
%    sn - Wavelet scattering network object
%    x  - Input image
%
%  Outputs:
%    features - Feature vector (1 x num_paths)
%  ========================================================================

function features = helperScatImages_mean(sn, x)
    % Compute scattering feature matrix
    smat = featureMatrix(sn, x);

    % Average across spatial dimensions (2:4 correspond to height, width, and channels)
    features = sum(smat, 2:4);

    % Transpose to row vector
    features = features';
end
