# Scattering Disk Visualization for Color Histopathology Images

Standalone implementation for generating publication-quality scattering disk visualizations from H&E stained histopathology images.


## Quick Start

1. Navigate to this folder in MATLAB:
   ```matlab
   cd /path/to/publication_scattering_disks/
   ```

2. Set up MATLAB paths:
   ```matlab
   setup_paths
   ```

3. Run the main script:
   ```matlab
   scat_disk_color_combined
   ```

## Key Parameters

Edit these parameters in `scat_disk_color_combined.m` (lines 53-68):

- `Nim` (line 53): Image size in pixels (default: 256, must be power of 2)
- `foptions.J` (line 56): Maximum wavelet scale (default: 2)
- `foptions.L` (line 57): Number of orientations (default: 6)
- `soptions.M` (line 60): Scattering order (default: 2)
- `combination_method` (line 68): RGB combination method
  - `'luminance'`: Weighted combination (recommended for H&E images)
  - `'average'`: Simple average
  - `'norm'`: L2 norm

## Input Data

Default input: `data/tumor_patch.tif`

To use your own image, modify line 114 in `scat_disk_color_combined.m`:
```matlab
img_color = imread('data/your_image.tif');
```

## Expected Outputs

The script generates four files in the current directory:
- `ScatteringDisks_Comparison.png` (600 DPI raster)

## File Structure
```
publication_scattering_disks/
├── scat_disk_color_combined.m  # Main script
├── setup_paths.m               # Path configuration
├── README.md                   # This file
├── data/                       # Input images
├── core/                       # Scattering transform core (5 files)
├── convolution/                # Convolution operations (7 files)
├── filters/                    # Wavelet filters (24 files)
│   └── selesnick/              # Selesnick filters (8 files)
├── utils/                      # Utilities (3 files)
└── helpers/                    # Visualization helpers (4 files)
```

## Troubleshooting

**Error: "Undefined function or variable"**
- Run `setup_paths` before executing the main script

**Error: "Unable to read file"**
- Ensure `data/tumor_patch.tif` exists
- Check file path in line 114

**Poor quality output**
- Increase image size `Nim` (must be power of 2: 128, 256, 512)
- Adjust wavelet parameters `J` and `L`
