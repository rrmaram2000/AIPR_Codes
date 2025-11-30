%% SETUP_PATHS - Configure MATLAB paths for standalone execution
% This script adds all necessary subdirectories to the MATLAB path
% for running scat_disk_color_combined.m
%
% USAGE:
%   Run this script once before executing scat_disk_color_combined.m
%   >> setup_paths
%   >> scat_disk_color_combined

function setup_paths()
    % Get the directory where this script is located
    script_dir = fileparts(mfilename('fullpath'));

    % Add all subdirectories to path
    addpath(genpath(script_dir));

    fprintf('========================================\n');
    fprintf('STANDALONE SCATTERING DISKS - PATH SETUP\n');
    fprintf('========================================\n');
    fprintf('Added to MATLAB path:\n');
    fprintf('  - %s (root)\n', script_dir);
    fprintf('  - %s/core\n', script_dir);
    fprintf('  - %s/convolution\n', script_dir);
    fprintf('  - %s/filters\n', script_dir);
    fprintf('  - %s/filters/selesnick\n', script_dir);
    fprintf('  - %s/utils\n', script_dir);
    fprintf('  - %s/helpers\n', script_dir);
    fprintf('  - %s/data\n', script_dir);
    fprintf('========================================\n');
    fprintf('Ready to run: scat_disk_color_combined\n');
    fprintf('========================================\n\n');
end
