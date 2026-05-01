# Elastic Vis: Anisotropic Elastic Property Visualization

`elastic-vis` is a robust Python tool for computing and visualizing anisotropic elastic properties (such as Young's Modulus and Shear Modulus) from a 6x6 stiffness tensor (Voigt notation). It provides high-resolution 2D polar plots and 3D surface representations designed for academic publication.

## Key Features

- **Elastic Properties Analysis**: Calculates Voigt, Reuss, and Hill averages for Bulk, Young's, and Shear moduli.
- **Stability Analysis**: Checks material mechanical stability using stiffness matrix eigenvalues.
- **High-Resolution Visualization**:
  - **2D Polar Plots**: Directional dependence in specific planes (XY, YZ, ZX) with perfect symmetry using bounded 1D optimization.
  - **3D Surface Plots**: Full 3D representations of directional properties with customizable perspective and high mesh density.
- **Publication-Quality Styling**: Rich support for custom fonts, colormaps, and layout settings via an intuitive JSON schema.
- **Robust Optimization**: Uses `scipy.optimize` with bounded search to ensure global extremes are accurately captured.
- **Versatile Input**: Supports stiffness matrices as strings, lists, or `numpy` arrays.
- **CLI Interface**: Powerful command-line tool for batch processing.

## Installation

You can install the package locally for development:

```bash
pip install -e .
```

## Usage

### Command Line Interface

The primary way to use the tool is via the `elastic-vis` command.

#### Basic Analysis
To see the elastic properties and stability analysis:
```bash
elastic-vis examples/matrix.txt
```

#### Generate Plots
To generate 2D and 3D plots for both Young's and Shear Modulus, and show them on screen:
```bash
elastic-vis examples/matrix.txt --plot --Young --Shear --2D --3D --show
```

#### Custom Styling
You can provide a JSON file to customize every aspect of the plots:
```bash
elastic-vis examples/matrix.txt --plot --style examples/style.json
```

### Python API

You can also use the library directly in your Python scripts. The `Elastic` class handles `numpy` arrays directly:

```python
import numpy as np
from elastic_vis import Elastic, plot_youngs_3d

# Using a numpy array
matrix = np.array([
    [20.49, 0.55, 7.42, 0, 0, 0],
    [0.55, 23.12, 2.17, 0, 0, 0],
    [7.42, 2.17, 18.61, 0, 0, 0],
    [0, 0, 0, 6.18, 0, 0],
    [0, 0, 0, 0, 12.08, 0],
    [0, 0, 0, 0, 0, 4.46]
])

material = Elastic(matrix)

# Get Hill averages
averages = material.averages()
print(f"Hill Bulk Modulus: {averages[2][0]:.2f} GPa")

# Generate and show a 3D plot
plot_youngs_3d(material, "my_material", show=True)
```

## Customizable Styling (`style.json`)

The `style.json` file provides granular control over the visuals. It is organized into three intuitive sections:

```json
{
  "general": {
    "font_family": "serif",
    "dpi": 600,
    "transparent": false
  },
  "plots_2d": {
    "figsize": [18, 7],
    "title_size": 20,
    "label_size": 14,
    "line_width": 3.0,
    "min_color": "#0072B2",
    "max_color": "#D55E00",
    "font_weight": "bold"
  },
  "plots_3d": {
    "figsize": [12, 10],
    "title_size": 22,
    "label_size": 16,
    "font_weight": "normal",
    "colormap": "magma",
    "view_elevation": 25,
    "view_azimuth": 45,
    "rcount": 200,
    "ccount": 200
  }
}
```

## Input Format

The input should be a 6x6 stiffness matrix in GPa (Voigt notation). It can be a simple `.txt` file:
```text
20.49     0.55     7.42        0        0        0  
0.55    23.12     2.17        0        0        0  
7.42     2.17    18.61        0        0        0  
0        0        0     6.18        0        0  
0        0        0        0    12.08        0  
0        0        0        0        0     4.46
```

## License

MIT
