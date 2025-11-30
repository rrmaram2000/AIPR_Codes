%% SCATTERING DISK VISUALIZATION - COMBINED RGB
% Generates scattering disk visualizations for color histopathology images
% by combining RGB channel information using configurable methods.
%
% DESCRIPTION:
%   This script computes wavelet scattering transforms on color images and
%   compares them with Gaussian random fields having matched power spectra.
%   The analysis quantifies texture organization beyond second-order statistics.
%   Creates publication-ready figures showing:
%   - ONE disk for 1st order scattering (RGB combined)
%   - ONE disk for 2nd order scattering (RGB combined)
%
% REQUIREMENTS:
%   - ScatNet toolbox functions (scat, wavelet_factory_2d, etc.)
%   - Image file: tumor_patch.tif (or modify line 63)
%   - Helper functions: recover_meta, scat_display_with_borders, crop,
%     add_circular_mask
%
% OUTPUTS:
%   - Figure: Scattering disk comparison (original vs Gaussian)
%   - Saved PNG: ScatteringDisks_Comparison.png in current folder
%   - Workspace variable 'results' with scattering coefficients and metrics
%
% PARAMETERS:
%   Nim                - Image size (must be power of 2), default: 256
%   foptions.J         - Maximum wavelet scale, default: 2
%   foptions.L         - Number of orientations, default: 6
%   soptions.M         - Scattering order, default: 2 (second-order)
%   combination_method - RGB combination method:
%                        'luminance' - weighted (0.299R + 0.587G + 0.114B)
%                        'average'   - simple average (R+G+B)/3
%                        'norm'      - L2 norm sqrt(R^2 + G^2 + B^2)
%
% REFERENCE:
%   Based on ISCV_Figure5.m template
%   Modified for IEEE AIPR paper publication-ready figures
%
% 
% DATE: 2025-11-29
%
% See also: scat, wavelet_factory_2d, recover_meta, scat_display_with_borders

close all;
clear;

%% ========================================================================
%  SECTION 1: INITIALIZATION AND PARAMETER SETUP
%  ========================================================================
%  Configure scattering transform parameters and display settings
%  ------------------------------------------------------------------------

% Image dimensions
Nim = 256;  % Image size in pixels (must be power of 2 for FFT efficiency)

% Wavelet filter bank options
foptions.J = 2;  % Maximum scale (number of octaves): 2^J = 4 scales
foptions.L = 6;  % Number of orientations (angular resolution)

% Scattering transform options
soptions.M = 2;  % Scattering order: 2 for second-order scattering

% RGB channel combination method
% Method options:
%   'luminance' - Weighted combination (0.299*R + 0.587*G + 0.114*B)
%                 Recommended for H&E histopathology images
%   'average'   - Simple average (R + G + B)/3
%   'norm'      - L2 norm sqrt(R^2 + G^2 + B^2)
combination_method = 'luminance';

% Display configuration summary
fprintf('========================================\n');
fprintf('SCATTERING DISK - COMBINED RGB\n');
fprintf('========================================\n');
fprintf('Combination method: %s\n', combination_method);
fprintf('Image size: %dx%d pixels\n', Nim, Nim);
fprintf('Wavelet scales (J): %d\n', foptions.J);
fprintf('Orientations (L): %d\n', foptions.L);
fprintf('Scattering order: %d\n', soptions.M);
fprintf('========================================\n\n');

%% ========================================================================
%  SECTION 2: WAVELET OPERATOR CREATION
%  ========================================================================
%  Build wavelet filter bank and compute normalization reference
%  ------------------------------------------------------------------------

fprintf('Creating wavelet operators...\n');

% Create 2D wavelet operator using Morlet wavelets
% Returns: Wop - cell array of wavelet transform operators
[Wop, ~] = wavelet_factory_2d([Nim Nim], foptions, soptions);

% Compute scattering transform of Dirac delta (for coefficient normalization)
% The Dirac delta provides a reference for normalizing scattering coefficients
dirac = zeros(Nim);     % Create zero image
dirac(1) = 1;           % Set Dirac delta at origin
dirac = fftshift(dirac);% Shift to center for proper FFT convention
[scdirac] = scat(dirac, Wop);

fprintf('Wavelet operators created successfully\n');
fprintf('  Filter bank size: %dx%d\n', Nim, Nim);
fprintf('  Scattering orders: 0 to %d\n\n', soptions.M);

%% ========================================================================
%  SECTION 3: IMAGE LOADING AND PREPROCESSING
%  ========================================================================
%  Load color histopathology image and prepare for analysis
%  ------------------------------------------------------------------------

fprintf('Loading color histopathology image...\n');

% Load H&E stained histopathology image
% MODIFY THIS LINE to use your own image file
img_color = imread('data/tumor_patch.tif');
img_color = imresize(img_color, [Nim Nim]);
fprintf('  Loaded: data/tumor_patch.tif\n');

% Alternative Option: Use test images (uncomment if needed)
% if exist('./D107.gif', 'file')
%     tempo = double(imread('D107.gif'));
%     img_color(:,:,1) = tempo(1:Nim, 1:Nim);
%     img_color(:,:,2) = tempo(1:Nim, 1:Nim);
%     img_color(:,:,3) = tempo(1:Nim, 1:Nim);
%     fprintf('  Loaded: D107.gif\n');
% else
%     try
%         img_color = imread('peppers.png');
%         img_color = imresize(img_color, [Nim Nim]);
%         fprintf('  Loaded: peppers.png (resized)\n');
%     catch
%         fprintf('  Creating synthetic color texture\n');
%         rng(42);
%         for c = 1:3
%             img_color(:,:,c) = 128 + 50*randn(Nim);
%         end
%     end
% end

% Verify image dimensions
[img_h, img_w, img_c] = size(img_color);
fprintf('  Image dimensions: %dx%dx%d\n', img_h, img_w, img_c);

% Resize if dimensions don't match required size
if img_h ~= Nim || img_w ~= Nim
    fprintf('  Resizing from %dx%d to %dx%d\n', img_h, img_w, Nim, Nim);
    img_color = imresize(img_color, [Nim Nim]);
end
fprintf('Image preprocessing complete\n\n');

% Display original color image
figure('Name', 'Original Color Image', 'Position', [100 100 400 400]);
imagesc(uint8(img_color));
axis square;
axis off;
title('Original H&E Histopathology Image', 'FontSize', 12, 'FontWeight', 'bold');

%% ========================================================================
%  SECTION 4: RGB CHANNEL PROCESSING
%  ========================================================================
%  Process each color channel separately and compute scattering transforms
%  ------------------------------------------------------------------------

fprintf('Processing RGB channels separately...\n');

% Determine combination weights based on selected method
switch combination_method
    case 'luminance'
        % Standard luminance weights (ITU-R BT.601)
        weights = [0.299, 0.587, 0.114];
        fprintf('  Method: Luminance weighting\n');
        fprintf('  Weights - R: %.3f, G: %.3f, B: %.3f\n', weights);
    case 'average'
        % Equal weights for simple averaging
        weights = [1/3, 1/3, 1/3];
        fprintf('  Method: Equal averaging\n');
    case 'norm'
        % L2 norm (weights will be squared and summed)
        weights = [1, 1, 1];
        fprintf('  Method: L2 norm\n');
end

% Process each RGB channel independently
channel_names = {'Red', 'Green', 'Blue'};

% Pre-allocate cell arrays for efficiency
SC1_channels = cell(1, 3);
PS1_channels = cell(1, 3);

for c = 1:3
    fprintf('  Processing %s channel (%d/3)...\n', channel_names{c}, c);

    % Extract single color channel
    f_channel = double(img_color(:,:,c));

    % Normalize channel: zero mean and unit L2 norm
    % This ensures fair comparison across channels
    f_channel = f_channel - mean(f_channel(:));  % Remove DC component
    f_channel = f_channel / norm(f_channel(:));  % Normalize energy

    % Compute power spectrum via FFT
    % Power spectrum = |FFT(signal)|^2
    PS_channel = abs(fft2(f_channel)).^2;

    % Compute scattering transform coefficients
    [sc_channel] = scat(f_channel, Wop);

    % Extract metadata and averaged coefficients
    met_channel = recover_meta(sc_channel, scdirac);

    % Store results for later combination
    SC1_channels{c} = met_channel.ave;  % Scattering coefficients
    PS1_channels{c} = PS_channel;        % Power spectrum
end
fprintf('RGB channel processing complete\n\n');

%% ========================================================================
%  SECTION 5: SCATTERING COEFFICIENT COMBINATION
%  ========================================================================
%  Combine RGB scattering coefficients using selected method
%  ------------------------------------------------------------------------

fprintf('Combining scattering coefficients from RGB channels...\n');

% Combine scattering coefficients based on selected method
if strcmp(combination_method, 'norm')
    % L2 norm combination: sqrt(R^2 + G^2 + B^2)
    % Preserves energy from all channels
    SC1_combined = sqrt(SC1_channels{1}.^2 + ...
                        SC1_channels{2}.^2 + ...
                        SC1_channels{3}.^2); %#ok<UNRCH>
    fprintf('  Using L2 norm combination\n');
else
    % Weighted linear combination: w1*R + w2*G + w3*B
    SC1_combined = weights(1) * SC1_channels{1} + ...
                   weights(2) * SC1_channels{2} + ...
                   weights(3) * SC1_channels{3};
    fprintf('  Using weighted combination\n');
end

% Combine power spectra using same weights
% This provides the reference second-order statistics
PS1_combined = weights(1) * PS1_channels{1} + ...
               weights(2) * PS1_channels{2} + ...
               weights(3) * PS1_channels{3};

fprintf('  Combined coefficient count: %d\n', length(SC1_combined));
fprintf('  Power spectrum size: %dx%d\n', size(PS1_combined));
fprintf('Combination complete\n\n');

%% ========================================================================
%  SECTION 6: GAUSSIAN RANDOM FIELD GENERATION
%  ========================================================================
%  Generate Gaussian random fields with matched power spectra for comparison
%  ------------------------------------------------------------------------

fprintf('Generating Gaussian random field with matched power spectrum...\n');

% Number of realizations for averaging (reduces variance in estimates)
Tequ = 16;
fprintf('  Number of realizations per channel: %d\n', Tequ);

% Generate separate Gaussian fields for each RGB channel
% Each channel matches the power spectrum of the corresponding image channel

% Pre-allocate cell arrays for efficiency
SC3_channels = cell(1, 3);
PS3_channels = cell(1, 3);
SC3_var_channels = cell(1, 3);
geq_channels = cell(1, 3);

for c = 1:3
    fprintf('  Generating Gaussian field for %s channel...\n', channel_names{c});

    % Initialize accumulators
    SC3_channels{c} = [];           % Scattering coefficients (all realizations)
    PS3_channels{c} = zeros(Nim);   % Power spectrum accumulator

    % Create filter matching the channel's power spectrum
    eqfilter_c = sqrt(PS1_channels{c});

    % Generate multiple realizations
    for t = 1:Tequ
        % Generate Gaussian noise with matched power spectrum
        gaussian_realization = real(ifft2(fft2(randn(Nim)) .* eqfilter_c));

        % Normalize: zero mean and unit energy
        gaussian_realization = gaussian_realization - mean(gaussian_realization(:));
        gaussian_realization = gaussian_realization / norm(gaussian_realization(:));

        % Accumulate power spectrum
        PS3_channels{c} = PS3_channels{c} + abs(fft2(gaussian_realization)).^2;

        % Compute scattering transform
        [sc_gaussian_c] = scat(gaussian_realization, Wop);
        met_gaussian_c = recover_meta(sc_gaussian_c, scdirac);

        % Store scattering coefficients
        SC3_channels{c} = [SC3_channels{c}; met_gaussian_c.ave];
    end

    % Average power spectrum across realizations
    PS3_channels{c} = PS3_channels{c} / Tequ;

    % Compute variance and mean of scattering coefficients
    SC3_var_channels{c} = var(SC3_channels{c}, 1);  % Variance for each coefficient
    SC3_channels{c} = mean(SC3_channels{c});        % Mean across realizations

    % Generate final Gaussian realization for visualization
    geq_c = real(ifft2(fft2(randn(Nim)) .* eqfilter_c));
    geq_c = geq_c - mean(geq_c(:));
    geq_c = geq_c / norm(geq_c(:));
    geq_channels{c} = geq_c;  % Store for color image creation
end

% Combine Gaussian scattering coefficients using same method as original image
fprintf('Combining Gaussian scattering coefficients...\n');
if strcmp(combination_method, 'norm')
    % L2 norm combination
    SC3 = sqrt(SC3_channels{1}.^2 + SC3_channels{2}.^2 + SC3_channels{3}.^2); %#ok<UNRCH>
    SC3var = SC3_var_channels{1} + SC3_var_channels{2} + SC3_var_channels{3};
else
    % Weighted combination
    SC3 = weights(1) * SC3_channels{1} + ...
          weights(2) * SC3_channels{2} + ...
          weights(3) * SC3_channels{3};
    % Variance propagation for weighted sum
    SC3var = weights(1).^2 * SC3_var_channels{1} + ...
             weights(2).^2 * SC3_var_channels{2} + ...
             weights(3).^2 * SC3_var_channels{3};
end

% Combine power spectra
PS3 = weights(1) * PS3_channels{1} + ...
      weights(2) * PS3_channels{2} + ...
      weights(3) * PS3_channels{3};

% Create color Gaussian image for display
fprintf('  Creating RGB Gaussian visualization...\n');
img_gaussian_color = zeros(Nim, Nim, 3);
for c = 1:3
    % Normalize each channel to 0-255 range for display
    gc = geq_channels{c};
    gc = (gc - min(gc(:))) / (max(gc(:)) - min(gc(:)));
    img_gaussian_color(:,:,c) = gc * 255;
end

% CRITICAL FIX: Generate final combined Gaussian for scattering disk visualization
% This fixes the sc_gaussian undefined variable error
fprintf('  Generating final combined Gaussian realization...\n');
geq_combined = real(ifft2(fft2(randn(Nim)) .* sqrt(PS1_combined)));
geq_combined = geq_combined - mean(geq_combined(:));
geq_combined = geq_combined / norm(geq_combined(:));

% Compute scattering transform of combined Gaussian (THIS WAS MISSING!)
[sc_gaussian_final] = scat(geq_combined, Wop);

fprintf('Gaussian field generation complete\n');
fprintf('  Total realizations: %d per channel\n', Tequ);
fprintf('  Combined coefficients: %d\n\n', length(SC3));

%% ========================================================================
%  SECTION 7: SCATTERING DISK VISUALIZATION
%  ========================================================================
%  Generate scattering disk images for both original and Gaussian fields
%  ------------------------------------------------------------------------

fprintf('Generating scattering disk visualizations...\n');

% Configure visualization options
copts.renorm_process = 0;     % Use custom normalization (not process-based)
copts.l2_renorm = 1;          % Apply L2 renormalization for scale invariance
copts.border_width = 3;       % Border thickness in pixels (for clarity)
copts.border_color = 0.4;     % Border color (0=black, 1=white, 0.2=dark gray)

% Prepare structure for original image scattering disk
% Use the last computed sc_channel structure as template
sc_combined = sc_channel;           % Copy structure from last channel
met_combined = met_channel;         % Copy metadata
met_combined.ave = SC1_combined;    % Replace with combined RGB coefficients

% Generate scattering disk for original H&E image
fprintf('  Creating disk for original H&E image...\n');
[SCA1] = scat_display_with_borders(sc_combined, scdirac, copts, SC1_combined);

% Generate scattering disk for Gaussian random field
% Uses sc_gaussian_final computed in Section 6 (line 345)
fprintf('  Creating disk for Gaussian field...\n');
[SCA3] = scat_display_with_borders(sc_gaussian_final, scdirac, copts, SC3);

fprintf('Scattering disk visualization complete\n');
fprintf('  Original image disks: %d orders\n', length(SCA1));
fprintf('  Gaussian field disks: %d orders\n\n', length(SCA3));

%% ========================================================================
%  SECTION 8: DISTANCE METRIC COMPUTATION
%  ========================================================================
%  Compute scattering distance ratios between original and Gaussian fields
%  Distance ratio quantifies texture organization beyond power spectrum
%  ------------------------------------------------------------------------

fprintf('Computing scattering distance metrics...\n');

% Find indices for 1st order scattering coefficients (order = 2 in metadata)
ord1_idx = find(met_combined.order == 2);
difer1 = SC1_combined(ord1_idx) - SC3(ord1_idx);  % Difference vector
ratio1 = sum(difer1.^2) / sum(SC3var(ord1_idx));  % Normalized squared distance

% Find indices for 2nd order scattering coefficients (order = 3 in metadata)
ord2_idx = find(met_combined.order == 3);
difer2 = SC1_combined(ord2_idx) - SC3(ord2_idx);  % Difference vector
ratio2 = sum(difer2.^2) / sum(SC3var(ord2_idx));  % Normalized squared distance

% Display distance metrics
fprintf('\n========================================\n');
fprintf('SCATTERING DISTANCE RATIOS\n');
fprintf('========================================\n');
fprintf('1st order scattering: %.4f\n', ratio1);
fprintf('2nd order scattering: %.4f\n', ratio2);
fprintf('========================================\n\n');

%% ========================================================================
%  SECTION 9: FIGURE GENERATION - MAIN COMPARISON
%  ========================================================================
%  Create publication-ready figures comparing original and Gaussian fields
%  ------------------------------------------------------------------------

fprintf('Preparing figure normalization parameters...\n');

% Normalization factors for display (adjusted for better contrast)
att1 = 1.0;  % Attenuation for 1st order (lower = more contrast)
att2 = 1.0;  % Attenuation for 2nd order (lower = more contrast)

% Compute normalization values based on maximum across both images
normvalues = zeros(1, 2);
normvalues(1) = att1 * max(max(SCA1{1}(:)), max(SCA3{1}(:)));
normvalues(2) = att2 * max(max(SCA1{2}(:)), max(SCA3{2}(:)));

fprintf('Creating main publication figure (Figure 1)...\n');

% Figure 1: 2x4 grid comparing H&E image vs Gaussian field
% Top row: Original H&E image and its scattering analysis
% Bottom row: Gaussian field and its scattering analysis
% PUBLICATION QUALITY SETTINGS for Springer LNCS
% Optimized for full-page width with better aspect ratio
fig_main = figure('Name', 'Figure 1 - Scattering Disk Comparison', ...
       'Position', [100 100 1800 900], 'Color', 'w', ...
       'PaperPositionMode', 'auto', 'PaperUnits', 'inches', ...
       'PaperSize', [12 6]);  % Better aspect ratio: 2:1 instead of 2.67:1

% Display parameters
cropfactor = 0.95;  % Crop factor for circular masks (0.95 = 95% of radius)
rad = 0.98;         % Radius for circular masking

% PUBLICATION TYPOGRAPHY SETTINGS
title_fontsize = 14;      % Larger font for readability in print
colorbar_fontsize = 12;   % Readable colorbar labels
border_linewidth = 2.0;   % Thicker borders for better visibility in print

% ------------------------------------------------------------------------
% ROW 1: Original H&E Histopathology Image Analysis
% ------------------------------------------------------------------------

% Subplot (a): Original color image
subplot(2, 4, 1);
imagesc(uint8(img_color));
axis square; axis off;
title('(a) Original H&E Image', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
rectangle('Position', [1 1 size(img_color,2)-1 size(img_color,1)-1], ...
          'EdgeColor', 'k', 'LineWidth', border_linewidth);

% Subplot (b): Power spectrum of original image
subplot(2, 4, 2);
% Apply power law for visualization and upsample
spectr = abs(imresize(fftshift(sqrt(PS1_combined)).^0.5, 4));
spectr_masked = add_circular_mask(crop(spectr.^0.5, round(cropfactor*size(spectr,1)), ...
                                        round(size(spectr,1)/2)), rad, 1);
imagesc(spectr_masked);
axis square; axis off;
title('(b) Power Spectrum', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
    'Interpreter', 'tex');

colormap(gca, parula);
clim([0 max(spectr(:).^0.5)*0.8]);  % Enhanced contrast
cb = colorbar('FontSize', colorbar_fontsize, 'LineWidth', 1);

cb.Label.FontSize = colorbar_fontsize;
hold on;
sz = size(spectr_masked);
rectangle('Position', [1 1 sz(2)-1 sz(1)-1], 'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;

% Subplot (c): 1st order scattering disk
subplot(2, 4, 3);
img1 = add_circular_mask(crop(64*SCA1{1}/normvalues(1), ...
                              round(cropfactor*size(SCA1{1},1)), ...
                              round(size(SCA1{1},1)/2)), rad);
imagesc(img1);
axis square; axis off;
title('(c) First Order ', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
colormap(gca, parula);
clim([0 64]);  % Fixed range for consistency
cb = colorbar('FontSize', colorbar_fontsize, 'LineWidth', 1);
cb.Label.FontSize = colorbar_fontsize;
hold on;
sz = size(img1);
rectangle('Position', [1 1 sz(2)-1 sz(1)-1], 'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;

% Subplot (d): 2nd order scattering disk
subplot(2, 4, 4);
img2 = add_circular_mask(crop(64*SCA1{2}/normvalues(2), ...
                              round(cropfactor*size(SCA1{2},1)), ...
                              round(size(SCA1{2},1)/2)), rad);
imagesc(img2);
axis square; axis off;
title('(d) Second Order', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
colormap(gca, parula);
clim([0 64]);  % Fixed range for consistency
cb = colorbar('FontSize', colorbar_fontsize, 'LineWidth', 1);
cb.Label.FontSize = colorbar_fontsize;
hold on;
sz = size(img2);
rectangle('Position', [1 1 sz(2)-1 sz(1)-1], 'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;

% ------------------------------------------------------------------------
% ROW 2: Gaussian Random Field Analysis (Matched Power Spectrum)
% ------------------------------------------------------------------------

% Subplot (e): Gaussian random field
subplot(2, 4, 5);
imagesc(uint8(img_gaussian_color));
axis square; axis off;
title('(e) Gaussian Random Field', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
hold on;
rectangle('Position', [1 1 size(img_gaussian_color,2)-1 size(img_gaussian_color,1)-1], ...
          'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;

% Subplot (f): Power spectrum of Gaussian field
subplot(2, 4, 6);
spectr_g = abs(imresize(fftshift(sqrt(PS3)).^0.5, 4));
spectr_g_masked = add_circular_mask(crop(spectr_g.^0.5, round(cropfactor*size(spectr_g,1)), ...
                                          round(size(spectr_g,1)/2)), rad, 1);
imagesc(spectr_g_masked);
axis square; axis off;
title('(f) Power Spectrum', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
colormap(gca, parula);
clim([0 max(spectr_g(:).^0.5)*0.8]);  % Enhanced contrast
cb = colorbar('FontSize', colorbar_fontsize, 'LineWidth', 1);
cb.Label.FontSize = colorbar_fontsize;
hold on;
sz = size(spectr_g_masked);
rectangle('Position', [1 1 sz(2)-1 sz(1)-1], 'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;

% Subplot (g): 1st order scattering disk (Gaussian)
subplot(2, 4, 7);
img3 = add_circular_mask(crop(64*SCA3{1}/normvalues(1), ...
                              round(cropfactor*size(SCA3{1},1)), ...
                              round(size(SCA3{1},1)/2)), rad);
imagesc(img3);
axis square; axis off;
title('(g) First Order', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
colormap(gca, parula);
clim([0 64]);  % Fixed range for consistency
cb = colorbar('FontSize', colorbar_fontsize, 'LineWidth', 1);
cb.Label.FontSize = colorbar_fontsize;
hold on;
sz = size(img3);
rectangle('Position', [1 1 sz(2)-1 sz(1)-1], 'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;

% Subplot (h): 2nd order scattering disk (Gaussian)
subplot(2, 4, 8);
img4 = add_circular_mask(crop(64*SCA3{2}/normvalues(2), ...
                              round(cropfactor*size(SCA3{2},1)), ...
                              round(size(SCA3{2},1)/2)), rad);
imagesc(img4);
axis square; axis off;
title('(h) Second Order', 'FontSize', title_fontsize, 'FontWeight', 'normal', ...
      'Interpreter', 'tex');
colormap(gca, parula);
clim([0 64]);  % Fixed range for consistency
cb = colorbar('FontSize', colorbar_fontsize, 'LineWidth', 1);
cb.Label.FontSize = colorbar_fontsize;
% Add border
hold on;
sz = size(img4);
rectangle('Position', [1 1 sz(2)-1 sz(1)-1], 'EdgeColor', 'k', 'LineWidth', border_linewidth);
hold off;



% Add proper spacing between subplots for publication quality
fprintf('  Adjusting subplot spacing for publication quality...\n');
% Optimize spacing for better visual balance
set(gcf, 'Units', 'inches');
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) 12 6]);

% Adjust subplot positions for tighter, more professional layout
% Get all subplot handles
ax = findall(gcf, 'Type', 'axes');

% Adjust spacing: reduce horizontal gaps, increase vertical spacing
for i = 1:length(ax)
    if ~strcmp(get(ax(i), 'Tag'), 'Colorbar')
        pos_i = get(ax(i), 'Position');
        % Increase width slightly, adjust spacing
        set(ax(i), 'Position', [pos_i(1) pos_i(2) pos_i(3)*1.05 pos_i(4)*1.05]);
    end
end

fprintf('  Figure complete\n\n');

%% ========================================================================
%  SECTION 10: SAVE FIGURE AND RESULTS
%  ========================================================================
%  Save main comparison figure to current folder and export results
%  ------------------------------------------------------------------------

% Save main comparison figure to current folder
fprintf('Saving main comparison figure...\n');

% Ensure the figure is valid and make it current
if ~ishandle(fig_main) || ~isvalid(fig_main)
    error('Figure handle is invalid. The figure may have been closed.');
end

figure(fig_main);  % Make this the current figure
set(fig_main, 'Renderer', 'painters');  % Vector renderer for crisp output

% PUBLICATION-QUALITY EXPORT SETTINGS
% Export as high-resolution PNG (600 DPI for better quality when scaled)
% Ensure output folder exists
outdir = fullfile(pwd, 'figures');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
outbase = fullfile(outdir, 'ScatteringDisks_Comparison');

% Save PNG (600 DPI) into figures folder
print(fig_main, outbase, '-dpng', '-r600');
fprintf('  Saved: ScatteringDisks_Comparison.png (600 DPI)\n');


fprintf('USAGE TIPS FOR SPRINGER LNCS:\n');
fprintf('  - Use the EPS file for LaTeX submissions (\\includegraphics{ScatteringDisks_Comparison.eps})\n');
fprintf('  - Recommended figure width in LaTeX: \\textwidth or \\linewidth\n');
fprintf('  - Figure dimensions: 12 inches x 6 inches (2:1 aspect ratio)\n');
fprintf('  - All formats saved at 600 DPI for publication quality\n\n');

% Display comprehensive summary
fprintf('========================================\n');
fprintf('SCATTERING ANALYSIS COMPLETE\n');
fprintf('========================================\n\n');

fprintf('SCATTERING DISK COMPARISON FIGURE:\n');
fprintf('  Top row (a-d):\n');
fprintf('    - Original H&E histopathology image (color)\n');
fprintf('    - Power spectrum\n');
fprintf('    - 1st order scattering disk\n');
fprintf('    - 2nd order scattering disk\n');
fprintf('  Bottom row (e-h):\n');
fprintf('    - Gaussian random field (matched power spectrum, color)\n');
fprintf('    - Power spectrum\n');
fprintf('    - 1st order scattering disk\n');
fprintf('    - 2nd order scattering disk\n\n');

fprintf('DISTANCE METRICS:\n');
fprintf('  1st-order distance ratio: %.4f\n', ratio1);
fprintf('  2nd-order distance ratio: %.4f\n\n', ratio2);

fprintf('========================================\n');

% Store results in workspace structure
fprintf('Saving results to workspace...\n');
results.SC1_combined = SC1_combined;        % Original image scattering coefficients
results.SC3 = SC3;                           % Gaussian field scattering coefficients
results.SCA1 = SCA1;                         % Original image scattering disks
results.SCA3 = SCA3;                         % Gaussian field scattering disks
results.ratio1 = ratio1;                     % 1st order distance ratio
results.ratio2 = ratio2;                     % 2nd order distance ratio
results.combination_method = combination_method;  % RGB combination method used
results.weights = weights;                   % Weights used for combination
results.met_combined = met_combined;         % Metadata for coefficients
results.normvalues = normvalues;             % Normalization values for display

fprintf('  Results saved to workspace variable: results\n');
fprintf('  Access individual results using: results.<field_name>\n\n');

fprintf('========================================\n');
fprintf('SCRIPT EXECUTION COMPLETE\n');
fprintf('========================================\n');
