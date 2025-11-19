%% ========================================================================
%  CRC TISSUE CLASSIFICATION USING WAVELET SCATTERING TRANSFORM
%  ------------------------------------------------------------------------
%  Streamlined pipeline that prepares the dataset, extracts scattering
%  features, normalizes them, trains an ECOC SVM, evaluates performance, and
%  saves both the figures and experiment artefacts.
%  ------------------------------------------------------------------------

clear; close all; clc;

%% Configuration and reproducibility
RANDOM_SEED = 100;
rng(RANDOM_SEED);
DATA_PATH = "/Users/ritishmaram/Desktop/Kather_texture_2016_image_tiles_5000";
CONFIG = buildConfig();
ensureFigureDir(CONFIG);
setupParallelPool();

%% Dataset loading and preview
fprintf('SECTION 2: Loading and exploring data...\n');
fullDatastore = loadDataset(DATA_PATH);
visualizeRandomSamples(fullDatastore, CONFIG);
fprintf('Data exploration complete.\n\n');

%% Split train/test sets
fprintf('SECTION 4: Splitting data into train and test sets...\n');
[trainImds, testImds, Imds] = splitDataset(DATA_PATH, CONFIG.train_ratio, RANDOM_SEED);
logSplitStatistics(trainImds, testImds);

%% Configure scattering network
sn = createScatteringNetwork(CONFIG);

%% Feature extraction + normalization
featureInfo = extractScatteringFeatures(sn, trainImds, testImds);
featureInfo = normalizeFeatureSets(featureInfo);

%% Train classifier
[classifier, training_time] = trainClassifier(featureInfo.train, trainImds.Labels, CONFIG);

%% Cross-validation
cvInfo = performCrossValidation(classifier, CONFIG.k_fold);

%% Test evaluation
evalInfo = evaluateClassifier(classifier, featureInfo.test, testImds.Labels);
metrics = computePerformanceMetrics(evalInfo.predicted_labels, testImds.Labels);

%% Visualizations
visualizeResults(metrics, testImds, evalInfo.predicted_labels, CONFIG);

%% Persist results
results = assembleResultsStruct(CONFIG, RANDOM_SEED, featureInfo, trainImds, testImds, ...
    classifier, evalInfo, metrics, training_time, cvInfo);
if CONFIG.save_results
    saveResultsStruct(results, CONFIG.output_file);
end

%% Summary
printSummary(Imds, trainImds, testImds, featureInfo.train, CONFIG, cvInfo, evalInfo, metrics, ...
    featureInfo.extraction_time, training_time);

fprintf('Analysis complete!\n');

%% ========================================================================
% Helper functions
% ========================================================================
function CONFIG = buildConfig()
    CONFIG = struct();
    CONFIG.train_ratio = 0.8;
    CONFIG.num_visualization_images = 20;
    CONFIG.k_fold = 10;
    CONFIG.image_size = [150 150];
    CONFIG.invariance_scale = 20;
    CONFIG.quality_factors = [1 1];
    CONFIG.num_rotations = [6 6];
    CONFIG.svm_kernel = 'polynomial';
    CONFIG.svm_poly_order = 3;
    CONFIG.svm_kernel_scale = 'auto';
    CONFIG.svm_box_constraint = 1;
    CONFIG.save_results = true;
    CONFIG.output_file = 'CRC_Classification_Results.mat';
    CONFIG.save_figures = true;
    CONFIG.figures_dir = fullfile(pwd, 'figures');
    CONFIG.figure_format = 'png';
end

function ensureFigureDir(CONFIG)
    if CONFIG.save_figures && exist(CONFIG.figures_dir, 'dir') ~= 7
        mkdir(CONFIG.figures_dir);
        fprintf('Created figure directory: %s\n', CONFIG.figures_dir);
    elseif CONFIG.save_figures
        fprintf('Figure directory already exists: %s\n', CONFIG.figures_dir);
    end
end

function setupParallelPool()
    fprintf('SECTION 3: Setting up parallel computing...\n');
    if isempty(gcp('nocreate'))
        parpool;
        fprintf('Parallel pool started.\n');
    else
        fprintf('Parallel pool already running.\n');
    end
    fprintf('\n');
end

function Imds = loadDataset(dataPath)
    Imds = imageDatastore(dataPath, 'IncludeSubFolders', true, ...
        'FileExtensions', '.tif', 'LabelSource', 'foldernames');
    fprintf('Total images loaded: %d\n', numel(Imds.Files));
    disp(unique(Imds.Labels));
end

function visualizeRandomSamples(Imds, CONFIG)
    fig_samples = figure('Name', 'Sample Images from Dataset', 'Position', [100 100 1200 800]);
    numImages = min(numel(Imds.Files), 5000);
    perm = randperm(numImages, CONFIG.num_visualization_images);
    for idx = 1:CONFIG.num_visualization_images
        subplot(4, 5, idx);
        imshow(imread(Imds.Files{perm(idx)}));
        label = char(Imds.Labels(perm(idx)));
        title(label(4:end), 'Interpreter', 'none');
    end
    sgtitle('Random Sample of CRC Tissue Images');
    saveFigure(fig_samples, 'Sample_Images.png', CONFIG);
end

function [trainImds, testImds, Imds] = splitDataset(dataPath, ratio, seed)
    rng(seed);
    Imds = imageDatastore(dataPath, 'IncludeSubFolders', true, ...
        'FileExtensions', '.tif', 'LabelSource', 'foldernames');
    Imds = shuffle(Imds);
    [trainImds, testImds] = splitEachLabel(Imds, ratio);
end

function logSplitStatistics(trainImds, testImds)
    fprintf('Training set distribution:\n');
    disp(countEachLabel(trainImds));
    fprintf('Testing set distribution:\n');
    disp(countEachLabel(testImds));
    fprintf('\n');
end

function sn = createScatteringNetwork(CONFIG)
    fprintf('SECTION 5: Creating wavelet scattering network...\n');
    sn = waveletScattering2('ImageSize', CONFIG.image_size, ...
        'InvarianceScale', CONFIG.invariance_scale, ...
        'QualityFactors', CONFIG.quality_factors, ...
        'NumRotations', CONFIG.num_rotations);
    [~, npaths] = paths(sn);
    fprintf('Scattering network configured with %d total paths.\n\n', sum(npaths));
end

function featureInfo = extractScatteringFeatures(sn, trainImds, testImds)
    fprintf('SECTION 6: Extracting wavelet scattering features...\n');
    tic;
    Trainf = gather(cellfun(@(x) helperScatImages_mean(sn, x), tall(trainImds), 'Uni', 0));
    Testf = gather(cellfun(@(x) helperScatImages_mean(sn, x), tall(testImds), 'Uni', 0));
    featureInfo.train = cat(1, Trainf{:});
    featureInfo.test = cat(1, Testf{:});
    featureInfo.extraction_time = toc;
    fprintf('Feature extraction complete in %.2f seconds.\n', featureInfo.extraction_time);
    fprintf('Training features: [%d, %d]\n', size(featureInfo.train));
    fprintf('Testing features: [%d, %d]\n\n', size(featureInfo.test));
end

function featureInfo = normalizeFeatureSets(featureInfo)
    mu = mean(featureInfo.train, 1);
    sigma = std(featureInfo.train, 0, 1);
    sigma(sigma == 0) = 1;
    featureInfo.train = (featureInfo.train - mu) ./ sigma;
    featureInfo.test = (featureInfo.test - mu) ./ sigma;
    featureInfo.normStats = struct('mean', mu, 'std', sigma);
    fprintf('Applied z-score normalization using training statistics.\n\n');
end

function [classifier, training_time] = trainClassifier(trainfeatures, trainLabels, CONFIG)
    fprintf('SECTION 7: Training SVM classifier...\n');
    t = templateSVM('KernelFunction', CONFIG.svm_kernel, ...
        'PolynomialOrder', CONFIG.svm_poly_order, ...
        'KernelScale', CONFIG.svm_kernel_scale, ...
        'BoxConstraint', CONFIG.svm_box_constraint, ...
        'Standardize', true);
    tic;
    classifier = fitcecoc(trainfeatures, trainLabels, 'Learners', t, ...
        'Coding', 'onevsall', 'Options', statset('UseParallel', true));
    training_time = toc;
    fprintf('Classifier training complete in %.2f seconds.\n\n', training_time);
end

function cvInfo = performCrossValidation(classifier, k)
    fprintf('SECTION 8: Performing %d-fold cross-validation...\n', k);
    tic;
    CVMdl = crossval(classifier, 'KFold', k);
    cvInfo.loss = kfoldLoss(CVMdl) * 100;
    cvInfo.accuracy = 100 - cvInfo.loss;
    cvInfo.cv_time = toc;
    if cvInfo.cv_time > 60
        fprintf('Cross-validation complete in %.2f minutes.\n', cvInfo.cv_time / 60);
    else
        fprintf('Cross-validation complete in %.2f seconds.\n', cvInfo.cv_time);
    end
    fprintf('%d-fold accuracy: %.2f%% | loss: %.2f%%\n\n', k, cvInfo.accuracy, cvInfo.loss);
end

function evalInfo = evaluateClassifier(classifier, testfeatures, testLabels)
    fprintf('SECTION 9: Testing classifier on held-out test set...\n');
    tic;
    evalInfo.predicted_labels = predict(classifier, testfeatures);
    evalInfo.prediction_time = toc;
    evalInfo.accuracy = mean(evalInfo.predicted_labels == testLabels) * 100;
    fprintf('Prediction complete in %.2f seconds.\n', evalInfo.prediction_time);
    fprintf('Test Set Accuracy: %.2f%% (%d/%d)\n\n', evalInfo.accuracy, ...
        sum(evalInfo.predicted_labels == testLabels), numel(testLabels));
end

function metrics = computePerformanceMetrics(predicted, actual)
    metrics.classes = unique(actual);
    metrics.cleanLabels = cellfun(@(x) x(4:end), cellstr(char(metrics.classes)), 'UniformOutput', false);
    metrics.confMat = confusionmat(actual, predicted);
    num_classes = numel(metrics.classes);
    metrics.precision = zeros(num_classes, 1);
    metrics.recall = zeros(num_classes, 1);
    metrics.f1 = zeros(num_classes, 1);
    for i = 1:num_classes
        TP = metrics.confMat(i, i);
        FP = sum(metrics.confMat(:, i)) - TP;
        FN = sum(metrics.confMat(i, :)) - TP;
        metrics.precision(i) = TP / max(TP + FP, eps);
        metrics.recall(i) = TP / max(TP + FN, eps);
        metrics.f1(i) = 2 * (metrics.precision(i) * metrics.recall(i)) / ...
            max(metrics.precision(i) + metrics.recall(i), eps);
    end
    metrics.avgPrecision = mean(metrics.precision) * 100;
    metrics.avgRecall = mean(metrics.recall) * 100;
    metrics.avgF1 = mean(metrics.f1) * 100;
end

function visualizeResults(metrics, testImds, predicted_labels, CONFIG)
    fprintf('SECTION 10: Generating result visualizations...\n');
    cleanTest = categorical(cellfun(@(x) x(4:end), cellstr(char(testImds.Labels)), 'UniformOutput', false));
    cleanPred = categorical(cellfun(@(x) x(4:end), cellstr(char(predicted_labels)), 'UniformOutput', false));

    fig_cm = figure('Name', 'Confusion Matrix', 'Position', [100 100 900 800]);
    cchart = confusionchart(cleanTest, cleanPred);
    cchart.Title = 'Confusion Matrix - SVM Classification Results';
    cchart.FontSize = 12;
    cchart.RowSummary = 'row-normalized';
    cchart.ColumnSummary = 'column-normalized';
    saveFigure(fig_cm, 'Confusion_Matrix.png', CONFIG);

    fig_perf = figure('Name', 'Per-Class Performance', 'Position', [100 100 1000 600]);
    bar_data = [metrics.precision * 100, metrics.recall * 100, metrics.f1 * 100];
    bar(bar_data);
    set(gca, 'XTickLabel', metrics.cleanLabels, 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Tissue Class', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Performance (%)', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Precision', 'Recall', 'F1-Score', 'Location', 'best');
    grid on;
    saveFigure(fig_perf, 'PerClass_Performance.png', CONFIG);
end

function results = assembleResultsStruct(CONFIG, RANDOM_SEED, featureInfo, trainImds, testImds, ...
    classifier, evalInfo, metrics, training_time, cvInfo)
    results = struct();
    results.config = CONFIG;
    results.random_seed = RANDOM_SEED;
    results.trainfeatures = featureInfo.train;
    results.testfeatures = featureInfo.test;
    results.normalization.mean = featureInfo.normStats.mean;
    results.normalization.std = featureInfo.normStats.std;
    results.trainImds = trainImds;
    results.testImds = testImds;
    results.classifier = classifier;
    results.predicted_labels = evalInfo.predicted_labels;
    results.actual_labels = testImds.Labels;
    results.accuracy = evalInfo.accuracy;
    results.crossval_accuracy = cvInfo.accuracy;
    results.crossval_loss = cvInfo.loss;
    results.confusion_matrix = metrics.confMat;
    results.precision = metrics.precision;
    results.recall = metrics.recall;
    results.f1_score = metrics.f1;
    results.extraction_time = featureInfo.extraction_time;
    results.training_time = training_time;
    results.cv_time = cvInfo.cv_time;
    results.prediction_time = evalInfo.prediction_time;
end

function saveResultsStruct(results, output_file)
    save(output_file, 'results', '-v7.3');
    fprintf('Results saved to: %s\n\n', output_file);
end

function printSummary(Imds, trainImds, testImds, trainfeatures, CONFIG, cvInfo, evalInfo, metrics, ...
    extraction_time, training_time)
    fprintf('========================================================================\n');
    fprintf('CLASSIFICATION SUMMARY\n');
    fprintf('========================================================================\n');
    fprintf('Dataset: Kather CRC Tissue Images\n');
    fprintf('Total images: %d (Train: %d, Test: %d)\n', numel(Imds.Files), ...
        numel(trainImds.Files), numel(testImds.Files));
    fprintf('Number of classes: %d\n', numel(metrics.classes));
    fprintf('Feature dimension: %d\n', size(trainfeatures, 2));
    fprintf('\nPerformance:\n');
    fprintf('  %d-fold Cross-validation accuracy: %.2f%%\n', CONFIG.k_fold, cvInfo.accuracy);
    fprintf('  Test set accuracy: %.2f%%\n', evalInfo.accuracy);
    fprintf('  Average precision: %.2f%%\n', metrics.avgPrecision);
    fprintf('  Average recall: %.2f%%\n', metrics.avgRecall);
    fprintf('  Average F1-score: %.2f%%\n', metrics.avgF1);
    fprintf('\nComputation time:\n');
    displayMinutesSeconds('Feature extraction', extraction_time);
    displayMinutesSeconds('Training', training_time);
    displayMinutesSeconds('Cross-validation', cvInfo.cv_time);
    displayMinutesSeconds('Prediction', evalInfo.prediction_time);
    fprintf('========================================================================\n');
end

function displayMinutesSeconds(label, duration)
    if duration > 60
        fprintf('  %s: %.2f minutes\n', label, duration / 60);
    else
        fprintf('  %s: %.2f seconds\n', label, duration);
    end
end

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

function features = helperScatImages_mean(sn, x)
    smat = featureMatrix(sn, x);
    features = sum(smat, 2:4)';
end
