import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from typing import NoReturn
from .core import Elastic, Elastic2D
from .plotting import (
    plot_youngs_2d, plot_youngs_3d,
    plot_shear_2d, plot_shear_3d,
    plot_poisson_2d, plot_poisson_3d,
    plot_lc_2d, plot_lc_3d,
)

def main() -> None:
    """The main entry point for the elastic-vis command-line interface.

    Parses command-line arguments, calculates elastic properties from an input
    stiffness matrix, and optionally generates visualization plots.
    """
    parser = argparse.ArgumentParser(description="Visualize anisotropic elastic properties from a stiffness tensor.")
    parser.add_argument("input_file", help="Path to the file containing the 6x6 stiffness matrix (txt, csv, or json).")
    parser.add_argument("--plot", action="store_true", help="Enable plot generation.")
    parser.add_argument("--Young", action="store_true", help="Generate Young's Modulus plots.")
    parser.add_argument("--Shear", action="store_true", help="Generate Shear Modulus plots.")
    parser.add_argument("--Poisson", action="store_true", help="Generate Poisson's Ratio plots.")
    parser.add_argument("--LC", action="store_true", help="Generate Linear Compressibility plots.")
    parser.add_argument("--2D", dest="plot_2d", action="store_true", help="Generate 2D polar plots.")
    parser.add_argument("--3D", dest="plot_3d", action="store_true", help="Generate 3D surface plots.")
    parser.add_argument("--output-prefix", help="Prefix for output image files.")
    parser.add_argument("--format", dest="output_format", choices=["png", "pdf", "svg"], default="png", help="Output file format for plots.")
    parser.add_argument("--style", help="Path to a JSON file containing matplotlib style settings.")
    parser.add_argument("--show", action="store_true", help="Show plots on screen.")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)

    style_config = {}
    if args.style:
        if not os.path.exists(args.style):
            print(f"Error: Style file {args.style} not found.")
            sys.exit(1)
        try:
            with open(args.style, 'r') as f:
                style_config = json.load(f)
        except Exception as e:
            print(f"Error parsing style JSON: {e}")
            sys.exit(1)

    try:
        with open(args.input_file, 'r') as f:
            content = f.read()

        # Determine if it is 2D or 3D
        temp_content = content.replace("|", " ").replace("(", " ").replace(")", " ").replace(",", " ")
        lines = [line for line in temp_content.split('\n') if line.strip()]
        if len(lines) == 3:
            material = Elastic2D(content)
        else:
            material = Elastic(content)
    except Exception as e:
        print(f"Error parsing matrix: {e}")
        sys.exit(1)

    # Print basic properties
    print(f"\n--- Elastic Properties ({'2D' if material.is2D() else '3D'}) ---")

    if material.is2D():
        # For 2D materials, we don't have Voigt/Reuss/Hill averages in the same way
        # But we can print some basic info or just the eigenvalues
        pass
    else:
        averages = material.averages()
        df = pd.DataFrame(averages,
                          columns=["Bulk (GPa)", "Young (GPa)", "Shear (GPa)", "Poisson"],
                          index=["Voigt", "Reuss", "Hill"])
        print(df)

    eigenvalues = material.eigenvalues()
    print(f"\nEigenvalues: {eigenvalues}")
    if np.all(eigenvalues > 0):
        print("Material is mechanically stable.")
    else:
        print("Material is mechanically unstable.")

    # Handle Plotting
    if args.plot:
        prefix = args.output_prefix if args.output_prefix else os.path.splitext(os.path.basename(args.input_file))[0]

        # Default to Young's if nothing specified but plot is true
        target_young = args.Young or (not (args.Shear or args.Poisson or args.LC))
        target_shear = args.Shear
        target_poisson = args.Poisson
        target_lc = args.LC

        # Default to both 2D and 3D if nothing specified (3D ignored for 2D materials)
        do_2d = args.plot_2d or (not args.plot_3d)
        do_3d = args.plot_3d or (not args.plot_2d)

        if material.is2D():
            # Special handling for 2D materials
            # For now, we reuse 2D plotting functions if they are compatible or update them
            from .plotting import (
                plot_youngs_2d_single, plot_shear_2d_single, plot_poisson_2d_single
            )
            if target_young and do_2d:
                plot_youngs_2d_single(material, prefix, style_config, show=args.show, output_format=args.output_format)
            if target_shear and do_2d:
                plot_shear_2d_single(material, prefix, style_config, show=args.show, output_format=args.output_format)
            if target_poisson and do_2d:
                plot_poisson_2d_single(material, prefix, style_config, show=args.show, output_format=args.output_format)
            if target_lc:
                print("Warning: Linear Compressibility plots not yet supported for 2D materials.")
        else:
            if target_young:
                if do_2d:
                    plot_youngs_2d(material, prefix, style_config, show=args.show, output_format=args.output_format)
                if do_3d:
                    plot_youngs_3d(material, prefix, style_config, show=args.show, output_format=args.output_format)

            if target_shear:
                if do_2d:
                    plot_shear_2d(material, prefix, style_config, show=args.show, output_format=args.output_format)
                if do_3d:
                    plot_shear_3d(material, prefix, style_config, show=args.show, output_format=args.output_format)

            if target_poisson:
                if do_2d:
                    plot_poisson_2d(material, prefix, style_config, show=args.show, output_format=args.output_format)
                if do_3d:
                    plot_poisson_3d(material, prefix, style_config, show=args.show, output_format=args.output_format)

            if target_lc:
                if do_2d:
                    plot_lc_2d(material, prefix, style_config, show=args.show, output_format=args.output_format)
                if do_3d:
                    plot_lc_3d(material, prefix, style_config, show=args.show, output_format=args.output_format)
if __name__ == "__main__":
    main()
