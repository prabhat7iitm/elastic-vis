# Elastic Vis: Anisotropic Elastic Property Visualization

`elastic-vis` is a Python tool for computing and visualizing anisotropic elastic properties (such as Young's Modulus and Shear Modulus) from a 6x6 stiffness tensor (Voigt notation). It provides both 2D polar plots and 3D surface representations.

## Features

- **Elastic Properties Analysis**: Calculates Voigt, Reuss, and Hill averages for Bulk, Young's, and Shear moduli.
- **Stability Analysis**: Checks material mechanical stability using stiffness matrix eigenvalues.
- **2D Visualization**: Generates polar plots of elastic properties in specific planes.
- **3D Visualization**: Generates 3D surface representations of the directional dependence of elastic moduli.
- **Customizable Styling**: Support for custom matplotlib styles via JSON configuration.
- **CLI Interface**: Easy-to-use command-line interface.

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
To generate 2D and 3D plots for Young's Modulus:
```bash
elastic-vis examples/matrix.txt --plot --Young --2D --3D
```

#### Custom Styling
You can provide a JSON file to customize the plot appearance (colors, fonts, etc.):
```bash
elastic-vis examples/matrix.txt --plot --style examples/style.json
```

### Options

| Argument | Description |
| --- | --- |
| `input_file` | Path to the 6x6 stiffness matrix (txt, csv, or json) |
| `--plot` | Enable plot generation |
| `--Young` | Generate Young's Modulus plots |
| `--Shear` | Generate Shear Modulus plots |
| `--2D` | Generate 2D polar plots |
| `--3D` | Generate 3D surface plots |
| `--style` | Path to a JSON matplotlib style file |
| `--output-prefix`| Prefix for output image files |

### Python API

You can also use the library directly in your Python scripts:

```python
from elastic_vis import Elastic, plot_youngs_3d

# Load matrix from string or list
matrix_data = """
20.49     0.55     7.42        0        0        0  
0.55    23.12     2.17        0        0        0  
7.42     2.17    18.61        0        0        0  
0        0        0     6.18        0        0  
0        0        0        0    12.08        0  
0        0        0        0        0     4.46
"""
material = Elastic(matrix_data)

# Get Hill averages
averages = material.averages()
print(f"Hill Bulk Modulus: {averages[2][0]} GPa")

# Generate a 3D plot
plot_youngs_3d(material, "my_material")
```

## Project Structure

- `elastic_vis/`: Core package containing the logic for calculations and plotting.
- `examples/`: Example stiffness matrices and styling configurations.
- `pyproject.toml`: Project metadata and dependency definitions.

## Input Format

The input should be a 6x6 stiffness matrix in GPa. Example (`examples/matrix.txt`):
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
