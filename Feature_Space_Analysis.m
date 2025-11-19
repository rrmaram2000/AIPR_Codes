%% =========================================================================
%  FEATURE SPACE ANALYSIS FOR CRC SCATTERING REPRESENTATIONS
%  -------------------------------------------------------------------------
%  This script consumes the saved CRC classification results
%  (CRC_Classification_Results.mat) and visualizes the learned wavelet
%  scattering feature space using a one-vs-all t-SNE strategy. Each tissue
%  class is contrasted against a balanced set of "rest" samples drawn only
%  from the held-out test split to provide clear, interpretable 2D
%  projections of class separability.
%  -------------------------------------------------------------------------

clear; clc; close all;

%% Configuration
analysisConfig = struct();
analysisConfig.results_file = 'CRC_Classification_Results.mat';
analysisConfig.standardize_features = true;
analysisConfig.random_seed = 7;
analysisConfig.samples_per_side = 125;          % positives + negatives drawn from test split
analysisConfig.tsne_perplexity = 20;            % tuned for ~250 samples (focus on local clusters)
analysisConfig.tsne_num_pca_components = 50;    % PCA pre-reduction inside t-SNE
analysisConfig.tsne_exaggeration = 12;          % stronger early exaggeration for cleaner separation
analysisConfig.tsne_learn_rate = 1000;
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
testFeatures = double(results.testfeatures);
allFeatures = testFeatures;

testLabels = results.testImds.Labels;
allLabels = testLabels;

if isempty(allFeatures)
    error('Feature matrix is empty. Ensure that feature extraction succeeded.');
end

% Clean label strings so plots are easier to read (remove numeric prefixes)
labelStrings = cellstr(allLabels);
cleanLabelStrings = regexprep(labelStrings, '^\d+_', '');
labelCats = categorical(cleanLabelStrings);
classes = categories(labelCats);
numClasses = numel(classes);

fprintf('Loaded %d test feature vectors across %d tissue classes.\n', size(allFeatures, 1), numClasses);

%% Optional standardization
if analysisConfig.standardize_features
    featuresProc = zscore(allFeatures);
else
    featuresProc = allFeatures;
end

fprintf('Feature standardization: %s\n', ternary(analysisConfig.standardize_features, 'enabled', 'disabled'));
fprintf('Launching one-vs-all t-SNE generation for %d classes.\n', numClasses);

%% One-vs-all t-SNE for every class (2D)
colors = lines(numClasses);

for classIdx = 1:numClasses
    targetClass = classes{classIdx};

    % Assemble balanced positives vs. negatives for the current one-vs-all view
    [ovrFeatures, binaryLabels, sampleStats] = assembleOvrDataset(featuresProc, labelCats, targetClass, analysisConfig);
    numSamples = size(ovrFeatures, 1);
    perplexity = computePerplexity(numSamples, analysisConfig.tsne_perplexity);

    fprintf('\nRunning t-SNE for %s vs rest (%d pos | %d rest | perplexity %d)\n', ...
        strrep(char(targetClass), '_', ' '), sampleStats.positiveCount, sampleStats.negativeCount, perplexity);

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
end

%% Summary
fprintf('\nFEATURE SPACE SUMMARY\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Samples analyzed : %d\n', size(featuresProc, 1));
fprintf('Feature dimension: %d\n', size(featuresProc, 2));
fprintf('Classes          : %s\n', strjoin(strrep(classes', '_', ' '), ', '));
fprintf('t-SNE views      : %d one-vs-all (2D, test split only).\n', numClasses);
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

function [ovrFeatures, binaryLabels, sampleStats] = assembleOvrDataset(features, labels, targetClass, config)
% assembleOvrDataset Balanced positive vs. negative sampling for one-vs-all
%   features       : standardized feature matrix (N x D)
%   labels         : categorical labels of length N
%   targetClass    : categorical scalar for the positive class
%   config         : struct with samples_per_side (requested positives + negatives)

    positiveIdx = find(labels == targetClass);
    negativeIdx = find(labels ~= targetClass);

    if isempty(positiveIdx)
        error('No positive samples found for class "%s".', char(targetClass));
    end

    if isempty(negativeIdx)
        error('No negative samples found for class "%s".', char(targetClass));
    end

    posCount = min(numel(positiveIdx), config.samples_per_side);
    negCount = min(numel(negativeIdx), config.samples_per_side);

    if posCount < 2 || negCount < 2
        error('Not enough samples to plot %s vs rest (pos=%d, neg=%d).', char(targetClass), posCount, negCount);
    end

    posSample = datasample(positiveIdx, posCount, 'Replace', false);
    negSample = balanceNegatives(negativeIdx, labels, targetClass, negCount);

    combinedIdx = [posSample; negSample];
    combinedLabels = categorical([repmat({strrep(char(targetClass), '_', ' ')}, numel(posSample), 1); ...
        repmat({['Non-', strrep(char(targetClass), '_', ' ')]}, numel(negSample), 1)]);

    % Shuffle to avoid ordering effects in visualization
    shuffleOrder = randperm(numel(combinedIdx));
    ovrFeatures = features(combinedIdx(shuffleOrder), :);
    binaryLabels = combinedLabels(shuffleOrder);

    sampleStats = struct('positiveCount', numel(posSample), 'negativeCount', numel(negSample));
end

function negSample = balanceNegatives(negativeIdx, labels, targetClass, requestedCount)
% balanceNegatives encourage a roughly even draw across non-target classes
    otherClasses = setdiff(categories(labels), {targetClass});
    perClassQuota = max(floor(requestedCount / numel(otherClasses)), 1);
    negSample = [];

    for i = 1:numel(otherClasses)
        classIdx = negativeIdx(labels(negativeIdx) == otherClasses{i});
        take = min(perClassQuota, numel(classIdx));
        if take > 0
            negSample = [negSample; datasample(classIdx, take, 'Replace', false)]; %#ok<AGROW>
        end
    end

    % If quota per class did not fill the requested slots, top up randomly
    remainingSlots = requestedCount - numel(negSample);
    if remainingSlots > 0
        remainingIdx = setdiff(negativeIdx, negSample);
        topUp = min(remainingSlots, numel(remainingIdx));
        negSample = [negSample; datasample(remainingIdx, topUp, 'Replace', false)]; %#ok<AGROW>
    end

    % Ensure we do not exceed available negatives
    negSample = negSample(1:min(requestedCount, numel(negSample)));
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
