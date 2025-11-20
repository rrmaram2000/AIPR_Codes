%% SCATTERING_PATH_ANALYSIS_FINAL
%   Generate a descriptive table of scattering paths for the CRC scattering
%   network, summarize scale/rotation metadata, optionally save results, and
%   provide guidance on typical queries.
%
%   This script performs the following steps:
%     1. Instantiate the scattering network with the CRC configuration.
%     2. Retrieve scattering paths and their associated wavelet metadata.
%     3. Build a consolidated table with per-path frequency/scale/rotation info.
%     4. Display summary statistics and usage examples.
%     5. Save the artifacts to the "wavelet_scattering_paths" folder.
%
%   The code avoids decorative output and focuses on reproducible results.

clear; clc; close all;

%% Configuration
cfg = struct();
cfg.imageSize = [150 150];
cfg.invarianceScale = 20;
cfg.qualityFactors = [1 1];
cfg.numRotations = [6 6];
cfg.saveFolder = fullfile(pwd, 'wavelet_scattering_paths');
cfg.supportThreshold = 0.05;

%% Create scattering network
sn = waveletScattering2('ImageSize', cfg.imageSize, ...
    'InvarianceScale', cfg.invarianceScale, ...
    'QualityFactors', cfg.qualityFactors, ...
    'NumRotations', cfg.numRotations);

%% Retrieve paths and filter bank metadata
[scatteringPaths, numPathsPerOrder] = paths(sn);
[phif, psifilters, centerFreqs] = filterbank(sn);
numRotations = cfg.numRotations(1);

%% Summarize paths per order
fprintf('Scattering paths per order:\n');
for orderIdx = 1:numel(numPathsPerOrder)
    fprintf('  Order %d: %d paths\n', orderIdx - 1, numPathsPerOrder(orderIdx));
end
fprintf('  Total: %d paths\n\n', sum(numPathsPerOrder));

%% Analyze each filter bank
waveletInfo = cell(numel(psifilters), 1);
for fb = 1:numel(psifilters)
    bank = psifilters{fb};
    numWavelets = size(bank, 3);

    cfx = centerFreqs{fb}(:, 1);
    cfy = centerFreqs{fb}(:, 2);
    freqMag = hypot(cfx, cfy);
    freqAngle = atan2(cfy, cfx);

    % Cluster by frequency magnitude and remap so highest freq -> scale 1
    [~, ~, rawScaleIdx] = uniquetol(freqMag, 1e-6);
    numScales = max(rawScaleIdx);
    avgFreqPerScale = accumarray(rawScaleIdx, freqMag, [], @mean);
    [~, descOrder] = sort(avgFreqPerScale, 'descend');
    scaleRemap = zeros(numScales, 1);
    scaleRemap(descOrder) = 1:numScales;
    scaleIdx = scaleRemap(rawScaleIdx);

    rotationIdx = mod((0:numWavelets-1)', numRotations);

    waveletInfo{fb} = struct( ...
        'numWavelets', numWavelets, ...
        'numScales', numScales, ...
        'numRotations', numRotations, ...
        'freqMag', freqMag, ...
        'freqAngle', freqAngle, ...
        'scaleIdx', scaleIdx, ...
        'rotationIdx', rotationIdx);

    fprintf('Filter bank %d: %d wavelets, %d scales, %d rotations\n', ...
        fb, numWavelets, numScales, numRotations);
end
fprintf('\n');

%% Build consolidated path table
pathTable = initializePathTable(numPathsPerOrder);

row = 1;
for orderIdx = 1:numel(scatteringPaths)
    orderTable = scatteringPaths{orderIdx};
    for ii = 1:height(orderTable)
        pathVec = orderTable.path(ii, :);
        pathTable.Order(row) = orderIdx - 1;
        pathTable.PathVector{row} = pathVec;

        % Filter bank 1 contribution (S1 and later)
        if numel(pathVec) >= 2
            waveIdx = pathVec(2);
            pathTable = assignWaveletMetadata(pathTable, row, 1, waveletInfo{1}, waveIdx);
        end

        % Filter bank 2 contribution (S2)
        if numel(pathVec) >= 3
            waveIdx = pathVec(3);
            pathTable = assignWaveletMetadata(pathTable, row, 2, waveletInfo{2}, waveIdx);
        end

        row = row + 1;
    end
end

fprintf('Path table created: %d rows × %d columns\n\n', ...
    height(pathTable), width(pathTable));

%% Summary display
showPathSummary(pathTable);
showVerification(pathTable);
showUsageGuide();

%% Save artefacts
saveScatteringPaths(cfg.saveFolder, scatteringPaths, pathTable, waveletInfo, ...
    numPathsPerOrder, centerFreqs);

%% ========================================================================
%  Local functions
% ========================================================================
function pathTable = initializePathTable(numPathsPerOrder)
    totalPaths = sum(numPathsPerOrder);
    pathTable = table();
    pathTable.GlobalIndex = (1:totalPaths)';
    pathTable.Order = zeros(totalPaths, 1);
    pathTable.PathVector = cell(totalPaths, 1);

    emptyCol = nan(totalPaths, 1);
    % Filter bank fields (1/2)
    pathTable.WaveletIdx_FB1 = emptyCol;
    pathTable.WaveletIdx_FB2 = emptyCol;
    pathTable.ScaleIdx_FB1 = emptyCol;
    pathTable.ScaleIdx_FB2 = emptyCol;
    pathTable.RotationIdx_FB1 = emptyCol;
    pathTable.RotationIdx_FB2 = emptyCol;
    pathTable.FreqMag_FB1 = emptyCol;
    pathTable.FreqMag_FB2 = emptyCol;
    pathTable.FreqAngle_FB1_deg = emptyCol;
    pathTable.FreqAngle_FB2_deg = emptyCol;
end

function pathTable = assignWaveletMetadata(pathTable, row, fbIdx, info, waveIdx)
%ASSIGNWAVELETMETADATA Populate filter bank columns for a given path row.
    switch fbIdx
        case 1
            pathTable.WaveletIdx_FB1(row) = waveIdx;
            pathTable.ScaleIdx_FB1(row) = info.scaleIdx(waveIdx);
            pathTable.RotationIdx_FB1(row) = info.rotationIdx(waveIdx);
            pathTable.FreqMag_FB1(row) = info.freqMag(waveIdx);
            pathTable.FreqAngle_FB1_deg(row) = info.freqAngle(waveIdx) * 180/pi;
        case 2
            pathTable.WaveletIdx_FB2(row) = waveIdx;
            pathTable.ScaleIdx_FB2(row) = info.scaleIdx(waveIdx);
            pathTable.RotationIdx_FB2(row) = info.rotationIdx(waveIdx);
            pathTable.FreqMag_FB2(row) = info.freqMag(waveIdx);
            pathTable.FreqAngle_FB2_deg(row) = info.freqAngle(waveIdx) * 180/pi;
    end
end

function showPathSummary(pathTable)
%SHOWPATHSUMMARY Display core table information.
    fprintf('Path table columns:\n');
    disp(pathTable.Properties.VariableNames');

    fprintf('\nFirst 10 paths:\n');
    disp(pathTable(1:min(10, height(pathTable)), ...
        {'GlobalIndex','Order','WaveletIdx_FB1','ScaleIdx_FB1','RotationIdx_FB1','FreqAngle_FB1_deg'}));

    fprintf('\nOrder 1 examples:\n');
    s1 = find(pathTable.Order == 1);
    disp(pathTable(s1(1:min(8, numel(s1))), ...
        {'GlobalIndex','WaveletIdx_FB1','ScaleIdx_FB1','RotationIdx_FB1','FreqMag_FB1','FreqAngle_FB1_deg'}));

    fprintf('\nOrder 2 examples:\n');
    s2 = find(pathTable.Order == 2);
    if ~isempty(s2)
        disp(pathTable(s2(1:min(8, numel(s2))), ...
            {'GlobalIndex','WaveletIdx_FB1','ScaleIdx_FB1', ...
             'WaveletIdx_FB2','ScaleIdx_FB2','FreqMag_FB1','FreqMag_FB2'}));
    end
end

function showVerification(pathTable)
%SHOWVERIFICATION Basic integrity checks.
    s1 = find(pathTable.Order == 1);
    s2 = find(pathTable.Order == 2);

    fprintf('\nVerification summary:\n');
    fprintf('  Order 1 paths with NaN scales: %d\n', sum(isnan(pathTable.ScaleIdx_FB1(s1))));
    if ~isempty(s2)
        fprintf('  Order 2 FB1 NaN scales: %d\n', sum(isnan(pathTable.ScaleIdx_FB1(s2))));
        fprintf('  Order 2 FB2 NaN scales: %d\n', sum(isnan(pathTable.ScaleIdx_FB2(s2))));

        s2Paths = pathTable(s2, :);
        freqCheck = s2Paths.FreqMag_FB2 < s2Paths.FreqMag_FB1;
        fprintf('  Frequency decrease satisfied: %d / %d\n', sum(freqCheck), numel(freqCheck));
    end
end

function showUsageGuide()
%SHOWUSAGEGUIDE Document sample queries.
    fprintf('\nUsage guide:\n');
    fprintf('  Inspect path index 15: pathTable(15, :)\n');
    fprintf('  All S1 paths at scale 2, rotation 0: \n');
    fprintf('      pathTable(pathTable.Order==1 & pathTable.ScaleIdx_FB1==2 & ...\n');
    fprintf('                pathTable.RotationIdx_FB1==0, :)\n');
    fprintf('  S2 paths fine->coarse (scale 2 -> 1): \n');
    fprintf('      pathTable(pathTable.Order==2 & pathTable.ScaleIdx_FB1==2 & ...\n');
    fprintf('                pathTable.ScaleIdx_FB2==1, :)\n');
end

function saveScatteringPaths(folder, scatteringPaths, pathTable, waveletInfo, numPathsPerOrder, centerFreqs)
%SAVESCATTERINGPATHS Persist results to the specified folder (MAT + CSV).
    if exist(folder, 'dir') ~= 7
        mkdir(folder);
    end
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    matFile = fullfile(folder, ['scattering_paths_' timestamp '.mat']);
    csvFile = fullfile(folder, ['scattering_paths_' timestamp '.csv']);
    save(matFile, 'scatteringPaths', 'pathTable', 'waveletInfo', 'numPathsPerOrder', 'centerFreqs');
    writetable(pathTable, csvFile);
    fprintf('\nResults saved to %s (MAT) and %s (CSV)\n', matFile, csvFile);
end

function cropped = cropCenter(img, targetSize)
%CROPCENTER Extract centered square crop with specified size.
    [h, w] = size(img);
    targetSize = min([targetSize, h, w]);
    r0 = floor((h - targetSize) / 2) + 1;
    c0 = floor((w - targetSize) / 2) + 1;
    cropped = img(r0:r0 + targetSize - 1, c0:c0 + targetSize - 1);
end

function [height, width] = estimateSupportSize(kernel, threshold)
%ESTIMATESUPPORTSIZE Bounding box of magnitudes >= threshold.
    if nargin < 2 || isempty(threshold)
        threshold = 0.05;
    end
    magnitude = abs(kernel);
    maxVal = max(magnitude(:));
    if maxVal == 0
        height = 0;
        width = 0;
        return;
    end
    mask = magnitude / maxVal >= threshold;
    [rows, cols] = find(mask);
    if isempty(rows)
        height = 0;
        width = 0;
        return;
    end
    height = max(rows) - min(rows) + 1;
    width = max(cols) - min(cols) + 1;
end

function spatialKernel = toSpatialKernel(freqKernel)
%TOSPATIALKERNEL Convert Fourier-domain filter to spatial domain.
    spatialKernel = ifftshift(ifft2(ifftshift(freqKernel)));
end
