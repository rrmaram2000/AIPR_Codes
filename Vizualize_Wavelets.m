% Vizualize_Wavelets Visualize 2D Morlet wavelet filters using waveletScattering2
%
% Syntax
%    Vizualize_Wavelets
%
% Description
%    Generates and visualizes 2D Morlet wavelet filters for multiple scales and
%    rotations using MATLAB's waveletScattering2. Displays colorized complex
%    wavelets (phase->hue, magnitude->lightness), their magnitudes, and the
%    scaling (low-pass) function.
%
% Inputs
%    None (parameters are set in the script: imageSize, J, numAngles)
%
% Outputs
%    Plots: colorized wavelets, magnitude maps, and scaling function. Prints
%    basic filter information to the command window and saves it to a text file.
%
% Requirements
%    Wavelet Toolbox (waveletScattering2), MATLAB R2020b or later recommended.
%    A MATLAB version of ploting filters based on the PyTorch based wavelet scattering kymatio.
%    For Python based routines, refer:  https://www.kymat.io/
%
% Examples
%    Run the script:
%       Vizualize_Wavelets
%
% See also
%    waveletScattering2, filterbank, ifft2, fftshift, imagesc, imshow

% Plot 2D Morlet Wavelet Filters
% ==============================
% This script generates and visualizes 2D Morlet wavelet filters
% for J = 0 (lambda_1 = 1) and J = 1 (lambda_1 = 2)
% with 6 angles evenly spaced between [0, pi)
%
% Based on the MATLAB waveletScattering2 framework
% See: https://www.mathworks.com/help/wavelet/ref/waveletscattering2.html

%% Initial parameters
% Image size
imageSize = [128, 128];

% Number of scales (J = 2 means we get j = 0 and j = 1)
J = 2;

% Number of angles (6 angles evenly spaced between [0, pi))
numAngles = 6;

%% Create wavelet scattering network
% Create the scattering network with specified parameters
sf = waveletScattering2('ImageSize', imageSize, ...
                        'NumRotations', [numAngles,numAngles], ...
                        'InvarianceScale', 5*2^J, ...
                        'QualityFactors', [1,1]);

%% Extract filter bank
% Get the wavelet filters from the scattering network
[phif, psifilters, f, filterparams] = filterbank(sf);

%% Prepare visualization
% The first filter bank contains wavelets at different scales and angles
psi = psifilters{1};  % Get first filter bank (3D array: M x N x L)

% Number of filters (should be J * numAngles)
numFilters = size(psi, 3);

% Angles in radians
angles = linspace(0, pi, numAngles + 1);
angles = angles(1:numAngles);  % Remove the endpoint (not including pi)

%% Display wavelets using complex phase coloring
% The colorization maps complex phase to hue and inverse magnitude to lightness
% Create publication-quality figure with larger size
fig = figure('Position', [100, 100, 2400, 1000], ...
             'Color', 'white', ...
             'PaperPositionMode', 'auto');

% Determine zoom region (central region of image for better visibility)
[M, N] = size(psi(:,:,1));
zoom_size = round(min(M, N) * 0.15);  % Zoom to 15% - much tighter crop
center = [round(M/2), round(N/2)];
row_range = (center(1) - zoom_size):(center(1) + zoom_size);
col_range = (center(2) - zoom_size):(center(2) + zoom_size);

for k = 1:numFilters
    % Get the current filter in frequency domain
    psi_f = psi(:, :, k);

    % Transform to spatial domain (following Python example approach)
    psi_spatial = ifft2(psi_f);
    psi_spatial = fftshift(psi_spatial);

    % Zoom in on the central region
    psi_zoomed = psi_spatial(row_range, col_range);

    % Determine scale j and angle index
    j = floor((k - 1) / numAngles);
    theta_idx = mod(k - 1, numAngles);

    % Calculate lambda_1 = 2^j
    lambda_1 = 2^j;

    % Create subplot
    subplot(J, numAngles, k);

    % Colorize and display the complex wavelet
    rgb = colorize(psi_zoomed);
    imshow(rgb);
    axis image on;

    % Add a thin rectangular frame around the image
    set(gca, 'XTick', [], 'YTick', [], ...
             'Box', 'on', ...
             'LineWidth', 1, ...
             'XColor', 'k', 'YColor', 'k');

    % Add title with lambda and angle in degrees - larger font for publication
    angle_deg = angles(theta_idx + 1) * 180 / pi;
    title(sprintf('\\lambda_1 = %d\n\\theta_1 = %d°', lambda_1, round(angle_deg)), ...
          'FontSize', 24, 'FontWeight', 'bold', 'Interpreter', 'tex');
end

% Improve subplot spacing for better readability
set(fig, 'Units', 'normalized');
% Adjust the spacing between subplots
h = get(fig, 'Children');
if ~isempty(h)
    % Get all axes
    all_axes = findall(fig, 'Type', 'axes');
    % Tighten the layout
    for ax_idx = 1:length(all_axes)
        pos = get(all_axes(ax_idx), 'Position');
        % Increase subplot size slightly for better visibility
        pos(3) = pos(3) * 1.1;  % width
        pos(4) = pos(4) * 1.1;  % height
        set(all_axes(ax_idx), 'Position', pos);
    end
end

%% Save figure and filter information
% Create output directory if it doesn't exist
output_dir = 'figures';
if ~isfolder(output_dir)
    mkdir(output_dir);
end

% Save the figure in multiple high-quality formats for publication

% 1. High-resolution PNG (300 DPI for print quality)
fig_filename_png = fullfile(output_dir, sprintf('wavelets_J%d_angles%d_highres.png', J, numAngles));
print(fig, fig_filename_png, '-dpng', '-r300');  % 300 DPI
fprintf('High-resolution PNG saved to: %s\n', fig_filename_png);

% 2. TIFF format (lossless, good for publications)
fig_filename_tiff = fullfile(output_dir, sprintf('wavelets_J%d_angles%d_highres.tif', J, numAngles));
print(fig, fig_filename_tiff, '-dtiff', '-r300');  % 300 DPI TIFF
fprintf('High-resolution TIFF saved to: %s\n', fig_filename_tiff);

% 3. EPS format (vector graphics, excellent for LaTeX/publication)
fig_filename_eps = fullfile(output_dir, sprintf('wavelets_J%d_angles%d.eps', J, numAngles));
print(fig, fig_filename_eps, '-depsc', '-r300');  % EPS with color
fprintf('EPS (vector) saved to: %s\n', fig_filename_eps);

% 4. PDF format (vector graphics, good for direct inclusion)
fig_filename_pdf = fullfile(output_dir, sprintf('wavelets_J%d_angles%d.pdf', J, numAngles));
exportgraphics(fig, fig_filename_pdf, 'ContentType', 'vector', 'Resolution', 300);
fprintf('PDF (vector) saved to: %s\n', fig_filename_pdf);


%% Print filter information
% Print summary metadata and sample filter parameters to the command window
fprintf('\n=== Filter Bank Information ===\n');
fprintf('Image Size: %d x %d\n', imageSize(1), imageSize(2));
fprintf('Number of Scales (J): %d\n', J);
fprintf('Number of Angles: %d\n', numAngles);
fprintf('Total Number of Wavelets: %d\n', numFilters);
fprintf('\nAngles (in radians): ');
fprintf('%.4f ', angles);
fprintf('\n\nAngles (as fractions of pi): ');
fprintf('%.4f*pi ', angles/pi);
fprintf('\n');

% Save filter information to text file
info_filename = fullfile(output_dir, sprintf('filter_info_J%d_angles%d.txt', J, numAngles));
fid = fopen(info_filename, 'w');

fprintf(fid, '=== Filter Bank Information ===\n');
fprintf(fid, 'Image Size: %d x %d\n', imageSize(1), imageSize(2));
fprintf(fid, 'Number of Scales (J): %d\n', J);
fprintf(fid, 'Number of Angles: %d\n', numAngles);
fprintf(fid, 'Total Number of Wavelets: %d\n', numFilters);
fprintf(fid, '\nAngles (in radians): ');
fprintf(fid, '%.4f ', angles);
fprintf(fid, '\n\nAngles (as fractions of pi): ');
fprintf(fid, '%.4f*pi ', angles/pi);
fprintf(fid, '\n');

if ~isempty(filterparams) && istable(filterparams{1})
    fprintf(fid, '\n=== Sample Filter Parameters ===\n');
    writetable(filterparams{1}(1:min(12, height(filterparams{1})), :), info_filename, 'WriteMode', 'append');
end

fclose(fid);
fprintf('Filter information saved to: %s\n', info_filename);

% Display filter parameters from the first few filters
fprintf('\n=== Sample Filter Parameters ===\n');
if ~isempty(filterparams) && istable(filterparams{1})
    disp(filterparams{1}(1:min(12, height(filterparams{1})), :));
end

%% Local helper functions

% Colorize complex wavelets
% Converts complex wavelet to RGB image where:
% - Hue represents complex phase
% - Lightness represents inverse magnitude
function rgb = colorize(im)
% COLORIZE Convert complex image to RGB representation using phase->hue and
% inverse magnitude->lightness mapping.
%
% Syntax
%    rgb = colorize(im)
%
% Inputs
%    im  - 2D complex array (wavelet in spatial domain)
%
% Outputs
%    rgb - M-by-N-by-3 RGB image with values in [0,1]
%
% Notes
%    The function normalizes the input by its maximum magnitude, maps the
%    phase to hue in [0,1), maps magnitude inversely to lightness and uses
%    a fixed saturation for vivid colors.
    % Normalize to have largest magnitude one
    max_val = max(abs(im(:)));
    if max_val > 0
        im = im / max_val;
    end

    % Use complex phase to determine hue (between 0 and 1)
    H = (angle(im) + pi) / (2 * pi);
    H = mod(H + 0.5, 1.0);

    % Use magnitude to determine lightness (inverse relationship)
    % Higher magnitude = darker (lower lightness)
    L = 1.0 ./ (1.0 + abs(im).^0.3);

    % Saturation (constant high saturation for vivid colors)
    S = 0.8 * ones(size(im));

    % Convert HLS to RGB (MATLAB uses HSV, so we need to adapt)
    % For HLS-like behavior, we'll use a custom conversion
    rgb = zeros([size(im), 3]);
    for i = 1:size(im, 1)
        for j = 1:size(im, 2)
            rgb(i, j, :) = hls_to_rgb(H(i,j), L(i,j), S(i,j));
        end
    end
end

% Helper function to convert HLS to RGB
function rgb = hls_to_rgb(h, l, s)
% HLS_TO_RGB Convert a single HLS triplet to RGB.
%
% Syntax
%    rgb = hls_to_rgb(h, l, s)
%
% Inputs
%    h - hue in [0,1)
%    l - lightness in [0,1]
%    s - saturation in [0,1]
%
% Outputs
%    rgb - 1x3 vector [r g b] in [0,1]
%
% Notes
%    This implementation mimics Python's colorsys.hls_to_rgb behavior for a
%    single scalar triplet.
    if s == 0
        rgb = [l, l, l];
        return;
    end

    if l <= 0.5
        m2 = l * (1 + s);
    else
        m2 = l + s - l * s;
    end
    m1 = 2 * l - m2;

    r = hue_to_rgb(m1, m2, h + 1/3);
    g = hue_to_rgb(m1, m2, h);
    b = hue_to_rgb(m1, m2, h - 1/3);

    rgb = [r, g, b];
end

% Helper function for HLS to RGB conversion
function val = hue_to_rgb(m1, m2, hue)
% HUE_TO_RGB Helper used by hls_to_rgb to interpolate a single color channel.
%
% Syntax
%    val = hue_to_rgb(m1, m2, hue)
%
% Inputs
%    m1, m2 - helper values computed from lightness and saturation
%    hue    - hue offset for the channel (may be outside [0,1], wrapped)
%
% Outputs
%    val - scalar channel value in [0,1]
    hue = mod(hue, 1.0);
    if hue < 1/6
        val = m1 + (m2 - m1) * hue * 6;
    elseif hue < 0.5
        val = m2;
    elseif hue < 2/3
        val = m1 + (m2 - m1) * (2/3 - hue) * 6;
    else
        val = m1;
    end
end