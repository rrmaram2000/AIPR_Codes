%% =========================================================================
%  VISUALIZE WAVELET SCATTERING FILTER BANK
%  -------------------------------------------------------------------------
%  Instantiates the CRC scattering network, retrieves the scaling + wavelet
%  filters, reports effective support, and exports zoomed spatial-domain
%  visualizations with consistent styling.
%  -------------------------------------------------------------------------

clear; clc; close all;

config = buildVizConfig();
rng(config.random_seed);
ensureFigureDir(config);

sn = createScatteringNetwork(config);
[filters, filterParams] = loadFilterBank(sn);
reportSupport(filters, config.support_threshold);

visualizeScaling(filters.phiFourier, config);
visualizeWaveletBanks(filters.psiFourier, filterParams, config);

fprintf('Wavelet filter visualizations generated under %s\n', config.figures_dir);

%% ========================================================================
%  Helper functions
% ========================================================================
function config = buildVizConfig()
    config = struct();
    config.image_size = [150 150];
    config.invariance_scale = 20;
    config.quality_factors = [1 1];
    config.num_rotations = [6 6];
    config.random_seed = 100;
    config.save_figures = true;
    config.figures_dir = fullfile(pwd, 'figures');
    config.figure_format = 'png';
    config.figure_bg_color = [1 1 1];
    config.colormap = buildLightRedBlue(256);
    config.display_window = 50;
    config.tile_output_size = 420;
    config.contrast_gamma = 0.6;
    config.energy_threshold = 0.02;
    config.apply_gaussian_taper = true;
    config.gaussian_taper_sigma = 0.28;
    config.apply_gaussian_smoothing = false;
    config.smoothing_sigma = 0.8;
    config.support_threshold = 0.05;
    config.scaling_render_mode = 'magnitude';
    config.wavelet_render_mode = 'signed';
    config.filterbank_indices = [];
    config.verbose = false;
end

function ensureFigureDir(config)
    if config.save_figures && exist(config.figures_dir, 'dir') ~= 7
        mkdir(config.figures_dir);
    end
end

function sn = createScatteringNetwork(config)
    sn = waveletScattering2('ImageSize', config.image_size, ...
        'InvarianceScale', config.invariance_scale, ...
        'QualityFactors', config.quality_factors, ...
        'NumRotations', config.num_rotations);
    [~, npaths] = paths(sn);
    fprintf('Scattering network configured with %d total paths.\n', sum(npaths));
end

function [filters, filterParams] = loadFilterBank(sn)
    phiFourier = [];
    psiFourier = {};
    filterParams = {};
    try
        [phiFourier, psiFourier, ~, filterParams] = filterbank(sn);
    catch
        try
            [phiFourier, psiFourier] = filterbank(sn);
        catch
            legacy = filters(sn);
            phiFourier = legacy.phi{1};
            psiFourier = legacy.psi;
        end
    end
    if ~iscell(psiFourier)
        psiFourier = {psiFourier};
    end
    filters = struct('phiFourier', phiFourier, 'psiFourier', {psiFourier});
end

function reportSupport(filters, threshold)
    fprintf('Scaling filter support: %d x %d pixels\n', size(filters.phiFourier));
    scalingKernel = toSpatialKernel(filters.phiFourier);
    [h, w] = estimateSupportSize(scalingKernel, threshold);
    fprintf('Scaling effective support (~threshold %.2f): %d x %d pixels\n', threshold, h, w);
end

function visualizeScaling(phiFourier, config)
    kernelSpatial = toSpatialKernel(phiFourier);
    scalingImg = prepareSpatialImage(kernelSpatial, config, config.scaling_render_mode);
    fig = figure('Name', 'Scaling Filter', 'Color', config.figure_bg_color, ...
        'Position', [100 100 620 520]);
    ax = axes(fig, 'Position', [0.08 0.08 0.85 0.85], 'Color', [1 1 1]);
    imagesc(ax, scalingImg);
    applyColormap(ax, config.scaling_render_mode, config.colormap);
    axis(ax, 'image');
    axis(ax, 'off');
    title(ax, 'Scaling Function (\phi)', 'Color', [0 0 0], 'FontSize', 16);
    cb = colorbar(ax, 'eastoutside');
    cb.Color = [0 0 0];
    cb.FontSize = 10;
    cb.Label.String = 'Normalized amplitude';
    saveFigure(fig, 'Scattering_Scaling_Filter', config);
end

function visualizeWaveletBanks(psiBanks, filterParams, config)
    availableBanks = numel(psiBanks);
    indices = config.filterbank_indices;
    if isempty(indices)
        indices = 1:availableBanks;
    else
        indices = indices(indices >= 1 & indices <= availableBanks);
    end
    if isempty(indices)
        warning('No valid filter bank indices selected. Nothing to visualize.');
        return;
    end

    for bankIdx = indices
        waveletVolume = ensureWaveletVolume(psiBanks{bankIdx});
        if isempty(waveletVolume)
            continue;
        end
        fprintf('Wavelet filter support (order %d): %d x %d pixels per orientation\n', ...
            bankIdx, size(waveletVolume, 1), size(waveletVolume, 2));
        sampleSpatial = toSpatialKernel(waveletVolume(:, :, 1));
        [h, w] = estimateSupportSize(sampleSpatial, config.support_threshold);
        fprintf('Wavelet effective support (order %d): %d x %d pixels\n', bankIdx, h, w);

        figTitle = sprintf('Wavelet Filters - Order %d', bankIdx);
        figName = sprintf('Scattering_Wavelets_Order%d', bankIdx);
        meta = deriveWaveletMetadata(bankIdx, waveletVolume, filterParams, config);
        figHandle = plotWaveletGrid(waveletVolume, meta, figTitle, config);
        saveFigure(figHandle, figName, config);
    end
end

function figHandle = plotWaveletGrid(waveletVolume, meta, figTitle, config)
    numScales = meta.numScales;
    numAngles = meta.numAngles;
    figHandle = figure('Name', figTitle, 'Color', config.figure_bg_color, ...
        'Position', [120 120 1200 800]);
    tl = tiledlayout(figHandle, numScales, numAngles, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, figTitle, 'Color', [0 0 0], 'FontWeight', 'bold', 'FontSize', 18);

    lastAxis = gobjects(numScales, 1);
    totalSlices = size(waveletVolume, 3);
    degSymbol = char(176);
    for scaleIdx = 1:numScales
        for angleIdx = 1:numAngles
            tileIndex = (scaleIdx - 1) * numAngles + angleIdx;
            ax = nexttile(tl, tileIndex);
            set(ax, 'Color', [1 1 1]);
            if tileIndex > totalSlices
                axis(ax, 'off');
                continue;
            end
            spatialKernel = toSpatialKernel(waveletVolume(:, :, tileIndex));
            tileImg = prepareSpatialImage(spatialKernel, config, config.wavelet_render_mode);
            imagesc(ax, tileImg);
            applyColormap(ax, config.wavelet_render_mode, config.colormap);
            axis(ax, 'image');
            axis(ax, 'off');
            labelStr = sprintf('J = %s, \\theta = %s%s', ...
                meta.scaleLabels{scaleIdx}, meta.angleLabels{angleIdx}, degSymbol);
            text(ax, 0.5, 0.95, labelStr, 'Units', 'normalized', ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                'Color', [0 0 0], 'FontSize', 10, 'FontWeight', 'bold', ...
                'Interpreter', 'tex');
            lastAxis(scaleIdx) = ax;
        end
    end

    for scaleIdx = 1:numScales
        ax = lastAxis(scaleIdx);
        if ~isgraphics(ax)
            continue;
        end
        cb = colorbar(ax, 'eastoutside');
        cb.Color = [0 0 0];
        cb.FontSize = 9;
        cb.Label.String = 'Normalized amplitude';
        cb.Position(3) = cb.Position(3) * 0.7;
        cb.Position(1) = cb.Position(1) + 0.02;
    end
end

function applyColormap(ax, mode, cmap)
    colormap(ax, cmap);
    if strcmpi(mode, 'signed')
        caxis(ax, [-1 1]);
    else
        caxis(ax, [0 1]);
    end
end

function spatialKernel = toSpatialKernel(freqKernel)
    spatialKernel = ifftshift(ifft2(ifftshift(freqKernel)));
end

function imgData = prepareSpatialImage(spatialKernel, config, mode)
    if nargin < 3 || isempty(mode)
        mode = 'signed';
    end
    switch lower(mode)
        case 'magnitude'
            imgData = abs(spatialKernel);
            maxVal = max(imgData(:));
            if maxVal > 0
                imgData = imgData ./ maxVal;
            end
        otherwise
            imgData = real(spatialKernel);
            maxAbsVal = max(abs(imgData(:)));
            if maxAbsVal > 0
                imgData = imgData ./ maxAbsVal;
            end
            imgData = imgData * 0.95;
    end

    if config.apply_gaussian_smoothing
        imgData = imgaussfilt(imgData, config.smoothing_sigma);
    end

    if ~isempty(config.energy_threshold)
        mask = abs(imgData) >= config.energy_threshold;
        imgData(~mask) = 0;
    end

    if config.apply_gaussian_taper
        sigma = config.gaussian_taper_sigma;
        [h, w] = size(imgData);
        [X, Y] = ndgrid(linspace(-1, 1, h), linspace(-1, 1, w));
        taper = exp(- (X.^2 + Y.^2) / (2 * sigma^2));
        imgData = imgData .* taper;
    end

    if ~isempty(config.display_window)
        imgData = cropCenter(imgData, config.display_window);
    end

    if ~isempty(config.tile_output_size)
        imgData = imresize(imgData, [config.tile_output_size config.tile_output_size], 'bicubic');
    end

    imgData = sign(imgData) .* abs(imgData) .^ config.contrast_gamma;
end

function [height, width] = estimateSupportSize(kernel, threshold)
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
    else
        height = max(rows) - min(rows) + 1;
        width = max(cols) - min(cols) + 1;
    end
end

function volume = ensureWaveletVolume(entry)
    if isempty(entry)
        volume = [];
    elseif isnumeric(entry)
        volume = entry;
    elseif iscell(entry)
        numericSlices = cellfun(@(c) isnumeric(c), entry);
        if all(numericSlices, 'all')
            volume = cat(3, entry{:});
        else
            error('Unsupported wavelet entry format.');
        end
    else
        error('Unsupported wavelet entry type: %s', class(entry));
    end
end

function meta = deriveWaveletMetadata(orderIdx, waveletVolume, filterParams, config)
    totalSlices = size(waveletVolume, 3);
    [angleLabels, numAngles] = deal([]);
    [scaleLabels, numScales] = deal([]);
    if ~isempty(filterParams) && numel(filterParams) >= orderIdx
        paramsEntry = filterParams{orderIdx};
        [angleLabels, numAngles] = extractAnglesFromParams(paramsEntry);
        [scaleLabels, numScales] = extractScalesFromParams(paramsEntry);
    end

    if isempty(numAngles)
        numAngles = min(extractConfigValue(config.num_rotations, orderIdx), size(waveletVolume, 3));
    end
    if isempty(numScales)
        numScales = max(ceil(totalSlices / numAngles), 1);
    end

    if isempty(angleLabels)
        angleVals = linspace(0, 360 - 360 / numAngles, numAngles);
        angleLabels = arrayfun(@(v) sprintf('%.0f', v), angleVals, 'UniformOutput', false);
    else
        angleLabels = arrayfun(@(v) sprintf('%.0f', mod(rad2deg(v), 360)), angleLabels, 'UniformOutput', false);
    end

    if isempty(scaleLabels)
        scaleLabels = arrayfun(@(v) sprintf('%d', v), numScales:-1:1, 'UniformOutput', false);
    else
        scaleLabels = arrayfun(@(v) sprintf('%.2f', v), scaleLabels, 'UniformOutput', false);
    end

    meta = struct('numAngles', numAngles, 'numScales', numScales, ...
        'angleLabels', {angleLabels}, 'scaleLabels', {scaleLabels});
end

function value = extractConfigValue(vector, idx)
    if isempty(vector)
        value = 6;
    else
        value = vector(min(idx, numel(vector)));
    end
end

function [angleLabels, numAngles] = extractAnglesFromParams(paramsEntry)
    angleLabels = [];
    numAngles = [];
    if istable(paramsEntry)
        names = lower(string(paramsEntry.Properties.VariableNames));
        idx = strcmp(names, "rotations");
        if any(idx)
            data = paramsEntry{:, idx};
            if iscell(data)
                data = data{1};
            end
            if isnumeric(data)
                angleLabels = data(:)';
                numAngles = numel(angleLabels);
            end
        end
    end
end

function [scaleLabels, numScales] = extractScalesFromParams(paramsEntry)
    scaleLabels = [];
    numScales = [];
    if istable(paramsEntry)
        names = lower(string(paramsEntry.Properties.VariableNames));
        idx = strcmp(names, "omegapsi");
        if any(idx)
            data = paramsEntry{:, idx};
            if iscell(data)
                data = data{1};
            end
            if isnumeric(data)
                scaleLabels = data(:)';
                numScales = numel(scaleLabels);
            end
        end
    end
end

function cropped = cropCenter(img, targetSize)
    [h, w] = size(img);
    targetSize = min([targetSize, h, w]);
    rowStart = floor((h - targetSize) / 2) + 1;
    colStart = floor((w - targetSize) / 2) + 1;
    cropped = img(rowStart:rowStart + targetSize - 1, colStart:colStart + targetSize - 1);
end

function saveFigure(figHandle, filename, config)
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
    exportPath = fullfile(config.figures_dir, filename);
    saveas(figHandle, exportPath);
    if config.verbose
        fprintf('Figure saved: %s\n', exportPath);
    end
end

function cmap = buildLightRedBlue(n)
    if nargin < 1
        n = 256;
    end
    half = floor(n/2);
    t1 = linspace(0, 1, half)';
    t2 = linspace(1, 0, n - half)';
    blueSide = [0.3 + 0.4*t1, 0.4 + 0.5*t1, 0.8 + 0.2*t1];
    redSide = [0.8 + 0.2*t2, 0.4 + 0.5*t2, 0.4 + 0.5*t2];
    cmap = [blueSide; redSide];
    cmap = max(min(cmap, 1), 0);
end
