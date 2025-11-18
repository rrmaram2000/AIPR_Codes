%% =========================================================================
%  FEATURE SPACE ANALYSIS FOR CRC SCATTERING REPRESENTATIONS
%  -------------------------------------------------------------------------
%  This standalone script consumes the saved CRC classification results
%  (CRC_Classification_Results.mat) and visualizes the learned wavelet
%  scattering feature space. It demonstrates how tissue classes separate
%  in high-dimensional space via PCA, t-SNE, and silhouette analysis to
%  reinforce interpretability claims made in the abstract.
%  -------------------------------------------------------------------------

clear; clc; close all;

%% Configuration
analysisConfig = struct();
analysisConfig.results_file = 'CRC_Classification_Results.mat';
analysisConfig.standardize_features = true;
analysisConfig.random_seed = 7;
analysisConfig.tsne_max_samples = 2000;
analysisConfig.tsne_num_dims = 2;
analysisConfig.tsne_perplexity = 30;
analysisConfig.tsne_num_pca_components = 50;
analysisConfig.tsne_exaggeration = 5;
analysisConfig.tsne_learn_rate = 500;
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
    [featuresProc, featureMu, featureSigma] = zscore(allFeatures);
else
    featuresProc = allFeatures;
    featureMu = mean(allFeatures, 1);
    featureSigma = std(allFeatures, 0, 1);
end

fprintf('Feature standardization: %s\n', ternary(analysisConfig.standardize_features, 'enabled', 'disabled'));

%% PCA analysis
numPcaComponents = min(10, size(featuresProc, 2));
[coeff, score, latent, ~, explained] = pca(featuresProc, 'NumComponents', numPcaComponents);

cumExplained = cumsum(explained);
fprintf('Top 3 PCA components explain %.2f%% of variance.\n', sum(explained(1:min(3, numel(explained)))));

colors = lines(numClasses);

figure_pca3d = figure('Name', 'PCA Scatter (3D)', 'Position', [100 100 1000 750]);
hold on;
for i = 1:numClasses
    idx = labelCats == classes{i};
    scatter3(score(idx, 1), score(idx, 2), score(idx, 3), 36, colors(i, :), 'filled', ...
        'DisplayName', strrep(char(classes{i}), '_', ' '));
end
grid on;
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');
title('3D PCA Projection of Scattering Features');
legend('Location', 'bestoutside');
view(45, 25);
hold off;
saveFigureLocal(figure_pca3d, 'PCA_3D_Scatter.png', analysisConfig);

figure_pcaExplained = figure('Name', 'PCA Explained Variance', 'Position', [200 150 900 500]);
plot(1:numel(cumExplained), cumExplained, '-o', 'LineWidth', 1.5);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Explained Variance of PCA');
grid on;
ylim([0 100]);
saveFigureLocal(figure_pcaExplained, 'PCA_Cumulative_Variance.png', analysisConfig);

%% t-SNE embedding
numSamples = min(analysisConfig.tsne_max_samples, size(featuresProc, 1));
if numSamples < size(featuresProc, 1)
    sampleIdx = randperm(size(featuresProc, 1), numSamples);
else
    sampleIdx = 1:size(featuresProc, 1);
end

tsneLabels = labelCats(sampleIdx);
perplexity = min(analysisConfig.tsne_perplexity, floor((numSamples - 1) / 3));
perplexity = max(perplexity, 5);  % ensure valid range

fprintf('Running t-SNE on %d samples with perplexity %d...\n', numel(sampleIdx), perplexity);

tsneEmbedding = tsne(featuresProc(sampleIdx, :), ...
    'NumDimensions', analysisConfig.tsne_num_dims, ...
    'Perplexity', perplexity, ...
    'Exaggeration', analysisConfig.tsne_exaggeration, ...
    'LearnRate', analysisConfig.tsne_learn_rate, ...
    'NumPCAComponents', min(analysisConfig.tsne_num_pca_components, size(featuresProc, 2)), ...
    'Standardize', false);

figure_tsne = figure('Name', 't-SNE Scatter', 'Position', [150 150 1000 750]);
hold on;
for i = 1:numClasses
    idx = tsneLabels == classes{i};
    scatter(tsneEmbedding(idx, 1), tsneEmbedding(idx, 2), 25, colors(i, :), 'filled', ...
        'DisplayName', strrep(char(classes{i}), '_', ' '));
end
hold off;
grid on;
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
title('t-SNE Projection of Scattering Features');
legend('Location', 'bestoutside');
saveFigureLocal(figure_tsne, 'tSNE_Scatter.png', analysisConfig);

%% Silhouette analysis (quantitative separability)
[silhouetteValues, silhFigure] = silhouette(featuresProc, labelCats);
meanSilhouette = mean(silhouetteValues);
title(sprintf('Silhouette Plot (Mean = %.3f)', meanSilhouette));
saveFigureLocal(silhFigure, 'Silhouette_Plot.png', analysisConfig);

fprintf('Mean silhouette coefficient: %.3f (higher indicates better separation).\n', meanSilhouette);

%% Summary
fprintf('\nFEATURE SPACE SUMMARY\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Samples analyzed : %d\n', size(featuresProc, 1));
fprintf('Feature dimension: %d\n', size(featuresProc, 2));
fprintf('Classes          : %s\n', strjoin(strrep(classes', '_', ' '), ', '));
fprintf('PCA (3 comps)    : %.2f%% variance captured.\n', sum(explained(1:min(3, numel(explained)))));
fprintf('t-SNE perplexity : %d with %d samples.\n', perplexity, numel(sampleIdx));
fprintf('Mean silhouette  : %.3f\n', meanSilhouette);
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
