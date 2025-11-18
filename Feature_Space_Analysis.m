%% =========================================================================
%  FEATURE SPACE ANALYSIS FOR CRC SCATTERING REPRESENTATIONS
%  -------------------------------------------------------------------------
%  This script consumes the saved CRC classification results
%  (CRC_Classification_Results.mat) and visualizes the learned wavelet
%  scattering feature space using a one-vs-all t-SNE strategy. Each tissue
%  class is contrasted against a balanced set of "rest" samples to provide
%  clear, interpretable 2D and 3D projections of class separability.
%  -------------------------------------------------------------------------

clear; clc; close all;

%% Configuration
analysisConfig = struct();
analysisConfig.results_file = 'CRC_Classification_Results.mat';
analysisConfig.standardize_features = true;
analysisConfig.random_seed = 7;
analysisConfig.tsne_max_samples = 400;          % per one-vs-all view (keeps plots readable)
analysisConfig.tsne_perplexity = 30;            % adjusted per sampled set automatically
analysisConfig.tsne_num_pca_components = 50;    % PCA pre-reduction inside t-SNE
analysisConfig.tsne_exaggeration = 5;
analysisConfig.tsne_learn_rate = 500;
analysisConfig.ovr_negatives_per_class = 25;    % cap of negatives drawn per non-target class
analysisConfig.save_figures = true;
analysisConfig.figure_format = 'png';
analysisConfig.figures_dir = fullfile(pwd, 'figures');

rng(analysisConfig.random_seed);

%% Load stored experimentation results
if exist(analysisConfig.results_file, 'file') ~= 2
    error('Results file "%s" not found. Please run the main classification script first.', ...
        analysisConfig.results_file);
end

S = load(analysisConfig.results_file, 'results');
results = S.results;

requiredFields = {'trainfeatures', 'testfeatures', 'trainImds', 'testImds'};
for i = 1:numel(requiredFields)
    if ~isfield(results, requiredFields{i})
        error('Results structure is missing required field "%s".', requiredFields{i});
    end
end

%% Assemble feature matrix and labels
trainFeatures = double(results.trainfeatures);
testFeatures = double(results.testfeatures);
allFeatures = [trainFeatures; testFeatures];

trainLabels = results.trainImds.Labels;
testLabels = results.testImds.Labels;
allLabels = [trainLabels; testLabels];

if isempty(allFeatures)
    error('Feature matrix is empty. Ensure that feature extraction succeeded.');
end

% Clean label strings so plots are easier to read (remove numeric prefixes)
labelStrings = cellstr(allLabels);
cleanLabelStrings = regexprep(labelStrings, '^\d+_', '');
labelCats = categorical(cleanLabelStrings);
classes = categories(labelCats);
numClasses = numel(classes);

fprintf('Loaded %d feature vectors across %d tissue classes.\n', size(allFeatures, 1), numClasses);

%% Optional standardization
if analysisConfig.standardize_features
    featuresProc = zscore(allFeatures);
else
    featuresProc = allFeatures;
end

fprintf('Feature standardization: %s\n', ternary(analysisConfig.standardize_features, 'enabled', 'disabled'));
fprintf('Launching one-vs-all t-SNE generation for %d classes.\n', numClasses);

%% One-vs-all t-SNE for every class (2D and 3D)
colors = lines(numClasses);

for classIdx = 1:numClasses
    targetClass = classes{classIdx};

    % Assemble balanced positives vs. negatives for the current one-vs-all view
    [ovrFeatures, binaryLabels] = assembleOvrDataset(featuresProc, labelCats, targetClass, analysisConfig);
    numSamples = size(ovrFeatures, 1);
    perplexity = computePerplexity(numSamples, analysisConfig.tsne_perplexity);

    fprintf('\nRunning t-SNE for %s vs rest (%d samples | perplexity %d)\n', ...
        strrep(char(targetClass), '_', ' '), numSamples, perplexity);

    % --- 2D projection ---
    tsne2d = tsne(ovrFeatures, ...
        'NumDimensions', 2, ...
        'Perplexity', perplexity, ...
        'Exaggeration', analysisConfig.tsne_exaggeration, ...
        'LearnRate', analysisConfig.tsne_learn_rate, ...
        'NumPCAComponents', min(analysisConfig.tsne_num_pca_components, size(ovrFeatures, 2)), ...
        'Standardize', false);

    fig2d = figure('Name', sprintf('t-SNE One-vs-All (2D) - %s', targetClass), ...
        'Position', [120 120 1000 750]);
    hold on;
    scatter(tsne2d(binaryLabels == binaryLabels(1), 1), tsne2d(binaryLabels == binaryLabels(1), 2), ...
        28, colors(classIdx, :), 'filled', 'DisplayName', strrep(char(targetClass), '_', ' '));
    scatter(tsne2d(binaryLabels ~= binaryLabels(1), 1), tsne2d(binaryLabels ~= binaryLabels(1), 2), ...
        28, [0.35 0.35 0.35], 'filled', 'DisplayName', ['Non-', strrep(char(targetClass), '_', ' ')]);
    hold off;
    grid on;
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    title(sprintf('One-vs-All t-SNE (2D): %s vs Rest', strrep(char(targetClass), '_', ' ')));
    legend('Location', 'bestoutside');
    saveFigureLocal(fig2d, sprintf('tSNE_OneVsAll_%s_2D', char(targetClass)), analysisConfig);

    % --- 3D projection ---
    tsne3d = tsne(ovrFeatures, ...
        'NumDimensions', 3, ...
        'Perplexity', perplexity, ...
        'Exaggeration', analysisConfig.tsne_exaggeration, ...
        'LearnRate', analysisConfig.tsne_learn_rate, ...
        'NumPCAComponents', min(analysisConfig.tsne_num_pca_components, size(ovrFeatures, 2)), ...
        'Standardize', false);

    fig3d = figure('Name', sprintf('t-SNE One-vs-All (3D) - %s', targetClass), ...
        'Position', [150 150 1000 750]);
    hold on;
    scatter3(tsne3d(binaryLabels == binaryLabels(1), 1), tsne3d(binaryLabels == binaryLabels(1), 2), tsne3d(binaryLabels == binaryLabels(1), 3), ...
        28, colors(classIdx, :), 'filled', 'DisplayName', strrep(char(targetClass), '_', ' '));
    scatter3(tsne3d(binaryLabels ~= binaryLabels(1), 1), tsne3d(binaryLabels ~= binaryLabels(1), 2), tsne3d(binaryLabels ~= binaryLabels(1), 3), ...
        28, [0.35 0.35 0.35], 'filled', 'DisplayName', ['Non-', strrep(char(targetClass), '_', ' ')]);
    hold off;
    grid on;
    xlabel('t-SNE Dimension 1');
    ylabel('t-SNE Dimension 2');
    zlabel('t-SNE Dimension 3');
    title(sprintf('One-vs-All t-SNE (3D): %s vs Rest', strrep(char(targetClass), '_', ' ')));
    legend('Location', 'bestoutside');
    view(45, 25);
    saveFigureLocal(fig3d, sprintf('tSNE_OneVsAll_%s_3D', char(targetClass)), analysisConfig);
end

%% Summary
fprintf('\nFEATURE SPACE SUMMARY\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Samples analyzed : %d\n', size(featuresProc, 1));
fprintf('Feature dimension: %d\n', size(featuresProc, 2));
fprintf('Classes          : %s\n', strjoin(strrep(classes', '_', ' '), ', '));
fprintf('t-SNE views      : %d one-vs-all (2D) and %d one-vs-all (3D).\n', numClasses, numClasses);
fprintf('Figures saved to : %s\n', analysisConfig.figures_dir);
fprintf('-------------------------------------------------------------\n');

%% Helper functions
function result = ternary(condition, trueString, falseString)
    if condition
        result = trueString;
    else
        result = falseString;
    end
end

function [ovrFeatures, binaryLabels] = assembleOvrDataset(features, labels, targetClass, config)
% assembleOvrDataset Balanced positive vs. negative sampling for one-vs-all
%   features    : standardized feature matrix (N x D)
%   labels      : categorical labels of length N
%   targetClass : categorical scalar for the positive class
%   config      : struct with tsne_max_samples and ovr_negatives_per_class

    positiveIdx = find(labels == targetClass);
    if isempty(positiveIdx)
        error('No positive samples found for class "%s".', char(targetClass));
    end

    otherClasses = setdiff(categories(labels), {targetClass});
    negativePool = [];

    % Collect a limited number of negatives from each non-target class
    for i = 1:numel(otherClasses)
        classIdx = find(labels == otherClasses{i});
        perClassSample = min(config.ovr_negatives_per_class, numel(classIdx));
        if perClassSample > 0
            negativePool = [negativePool; datasample(classIdx, perClassSample, 'Replace', false)]; %#ok<AGROW>
        end
    end

    % If needed, top up with remaining negatives to balance positives
    remainingNegatives = setdiff(find(labels ~= targetClass), negativePool);
    if numel(negativePool) < numel(positiveIdx) && ~isempty(remainingNegatives)
        fillCount = min(numel(positiveIdx) - numel(negativePool), numel(remainingNegatives));
        negativePool = [negativePool; datasample(remainingNegatives, fillCount, 'Replace', false)]; %#ok<AGROW>
    end

    if isempty(negativePool)
        error('No negative samples found for class "%s".', char(targetClass));
    end

    % Balance counts and enforce overall sample cap
    maxPerSide = floor(config.tsne_max_samples / 2);
    posCount = min(numel(positiveIdx), maxPerSide);
    negCount = min(numel(negativePool), maxPerSide);
    balancedCount = min(posCount, negCount);

    if balancedCount < 2
        error('Not enough balanced samples to plot %s vs rest (have %d).', char(targetClass), balancedCount);
    end

    posSample = datasample(positiveIdx, balancedCount, 'Replace', false);
    negSample = datasample(negativePool, balancedCount, 'Replace', false);

    combinedIdx = [posSample; negSample];
    combinedLabels = categorical([repmat({strrep(char(targetClass), '_', ' ')}, balancedCount, 1); ...
        repmat({['Non-', strrep(char(targetClass), '_', ' ')]}, balancedCount, 1)]);

    % Shuffle to avoid ordering effects in visualization
    shuffleOrder = randperm(numel(combinedIdx));
    ovrFeatures = features(combinedIdx(shuffleOrder), :);
    binaryLabels = combinedLabels(shuffleOrder);
end

function perplexity = computePerplexity(numSamples, basePerplexity)
% computePerplexity Clamp perplexity to valid bounds for the sampled set
    maxAllowed = max(floor((numSamples - 1) / 3), 1);
    perplexity = min(basePerplexity, maxAllowed);
    perplexity = max(perplexity, min(5, maxAllowed));
end

function saveFigureLocal(figHandle, filename, config)
    if ~config.save_figures || isempty(figHandle)
        return;
    end

    if exist(config.figures_dir, 'dir') ~= 7
        mkdir(config.figures_dir);
    end

    [~, ~, ext] = fileparts(filename);
    if isempty(ext)
        ext = ['.', config.figure_format];
        filename = [filename, ext];
    end

    fullPath = fullfile(config.figures_dir, filename);
    saveas(figHandle, fullPath);
    fprintf('Figure saved: %s\n', fullPath);
end
