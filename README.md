# Elastic Vis: Anisotropic Elastic Property Visualization

`elastic-vis` is a Python tool for computing and visualizing anisotropic elastic properties from an exact 6x6 stiffness tensor in Voigt notation. It provides high-resolution 2D polar plots and 3D surface representations designed for academic publication.

## Key Features

- **Elastic Properties Analysis**: Calculates Voigt, Reuss, and Hill averages for Bulk, Young's, and Shear moduli.
- **Stability Analysis**: Checks material mechanical stability using stiffness matrix eigenvalues.
- **Directional Properties**: Computes Young's modulus, shear modulus, Poisson's ratio, and linear compressibility from the compliance tensor.
- **High-Resolution Visualization**:
  - **2D Polar Plots**: Directional dependence in specific planes (XY, YZ, ZX) with bounded 1D optimization.
  - **3D Surface Plots**: Full 3D representations of directional properties with customizable perspective and high mesh density.
- **Publication-Quality Styling**: Rich support for custom fonts, colormaps, and layout settings via an intuitive JSON schema.
- **Robust Optimization**: Uses `scipy.optimize` with bounded search to ensure global extremes are accurately captured.
- **Versatile Input**: Supports stiffness matrices as strings, lists, or `numpy` arrays, but requires an actual 6x6 matrix.
- **CLI Interface**: Command-line tool for batch processing with `png`, `pdf`, and `svg` output.

## Installation

You can install the package by cloning and performing:

```bash
pip install .
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

To save plots as PDF instead of PNG:
```bash
elastic-vis examples/matrix.txt --plot --Young --Shear --2D --3D --format pdf
```

To save plots as SVG:
```bash
elastic-vis examples/matrix.txt --plot --Young --Shear --2D --3D --format svg
```

To generate Poisson's ratio plots:
```bash
elastic-vis examples/matrix.txt --plot --Poisson --2D --3D --format pdf
```

To generate linear compressibility plots:
```bash
elastic-vis examples/matrix.txt --plot --LC --2D --3D --format pdf
```

Plot conventions:
- Polar plots now place `0°` at the east/right side of the figure.
- The XY plane slice is computed with `theta = pi/2` and `phi` varying.
- The CLI writes filenames using the chosen format, for example `examples/example_Youngs_Modulus_3D.pdf`.

Available plot flags:
- `--Young`
- `--Shear`
- `--Poisson`
- `--LC`
- `--2D`
- `--3D`
- `--format png|pdf|svg`

#### Custom Styling
You can provide a JSON file to customize every aspect of the plots:
```bash
elastic-vis examples/matrix.txt --plot --style examples/style.json
```

### Python API

You can also use the library directly in your Python scripts. The package exposes three main classes:

- `Elastic`: 3D stiffness tensors in Voigt notation
- `ElasticOrtho`: optimized 3D calculations for orthorhombic tensors
- `Elastic2D`: 2D stiffness tensors

The `Elastic` class handles `numpy` arrays directly:

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

The most useful `Elastic` methods are:

```python
# Scalar properties
material.averages()      # Voigt, Reuss, Hill averages
material.eigenvalues()   # Stiffness matrix eigenvalues

# Directional Young's modulus
material.Young([theta, phi])

# Linear compressibility
material.LC([theta, phi])
material.LC_2(theta, phi)

# Shear modulus
material.shear([theta, phi, chi])
material.shear2D([theta, phi])     # min/max over chi
material.shear3D(theta, phi)       # min/max over chi + best chi values

# Poisson's ratio
material.Poisson([theta, phi, chi])
material.Poisson2D([theta, phi])   # min/max over chi
material.Poisson3D(theta, phi)     # min/max over chi + best chi values
```

For 2D materials, `Elastic2D` provides:

```python
material = Elastic2D(matrix_3x3)
material.Young(theta)
material.shear(theta)
material.Poisson(theta)
material.eigenvalues()
```

If your tensor is orthorhombic, `ElasticOrtho` uses a faster closed-form implementation for selected directional calculations.

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

The input must be an exact 6x6 stiffness matrix in GPa (Voigt notation). If the file is not 6x6, `elastic-vis` exits with an error instead of trying to infer the shape or pad triangular input.

It can be a simple `.txt` file:
```text
20.49     0.55     7.42        0        0        0  
0.55    23.12     2.17        0        0        0  
7.42     2.17    18.61        0        0        0  
0        0        0     6.18        0        0  
0        0        0        0    12.08        0  
0        0        0        0        0     4.46
```

## Example Outputs

The repository includes sample PDF outputs in `examples/`, generated with:

```bash
elastic-vis examples/matrix.txt --plot --Young --Shear --2D --3D --format pdf --output-prefix examples/example
```

Additional sample PDFs are generated with:

```bash
elastic-vis examples/matrix.txt --plot --Poisson --LC --2D --3D --format pdf --output-prefix examples/example
```

This produces:

- `examples/example_Youngs_Modulus_Polar_2D.pdf`
- `examples/example_Youngs_Modulus_3D.pdf`
- `examples/example_Shear_Modulus_Polar_2D.pdf`
- `examples/example_Shear_Modulus_3D.pdf`
- `examples/example_Poisson_Ratio_Polar_2D.pdf`
- `examples/example_Poisson_Ratio_3D.pdf`
- `examples/example_Linear_Compressibility_Polar_2D.pdf`
- `examples/example_Linear_Compressibility_3D.pdf`

## License

MIT
