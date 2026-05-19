import pytest
import os
import subprocess
import sys

def run_cli(args):
    # Get the path to the elastic-vis script or run via python -m
    # Since it's installed in editable mode, we can just run the command
    cmd = ["elastic-vis"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_cli_help():
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "Visualize anisotropic elastic properties" in result.stdout

def test_cli_3d_basic(tmp_path):
    mat_file = tmp_path / "matrix.txt"
    mat_file.write_text("100 20 20 0 0 0\n20 100 20 0 0 0\n20 20 100 0 0 0\n0 0 0 40 0 0\n0 0 0 0 40 0\n0 0 0 0 0 40")
    
    result = run_cli([str(mat_file)])
    assert result.returncode == 0
    assert "Elastic Properties (3D)" in result.stdout
    assert "Material is mechanically stable" in result.stdout

def test_cli_2d_basic(tmp_path):
    mat_file = tmp_path / "matrix_2d.txt"
    mat_file.write_text("100 20 0\n20 100 0\n0 0 40")
    
    result = run_cli([str(mat_file)])
    assert result.returncode == 0
    assert "Elastic Properties (2D)" in result.stdout

def test_cli_plotting_3d(tmp_path):
    # Change directory to tmp_path to avoid polluting project root
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mat_file = tmp_path / "matrix.txt"
        mat_file.write_text("100 20 20 0 0 0\n20 100 20 0 0 0\n20 20 100 0 0 0\n0 0 0 40 0 0\n0 0 0 0 40 0\n0 0 0 0 0 40")
        
        result = run_cli([str(mat_file), "--plot", "--Young", "--2D", "--3D"])
        assert result.returncode == 0
        assert os.path.exists("matrix_Youngs_Modulus_Polar_2D.png")
        assert os.path.exists("matrix_Youngs_Modulus_3D.png")
    finally:
        os.chdir(old_cwd)

def test_cli_plotting_2d(tmp_path):
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        mat_file = tmp_path / "matrix_2d.txt"
        mat_file.write_text("100 20 0\n20 100 0\n0 0 40")
        
        result = run_cli([str(mat_file), "--plot", "--Young", "--2D"])
        assert result.returncode == 0
        assert os.path.exists("matrix_2d_Youngs_Modulus_2D.png")
    finally:
        os.chdir(old_cwd)
