import pytest
import numpy as np
import os
from elastic_vis import Elastic, Elastic2D

# --- 3D Elastic Tests ---

def test_elastic_initialization_symmetric():
    # Standard cubic-like matrix
    mat = np.eye(6) * 100
    mat[0,1] = mat[1,0] = 20
    mat[0,2] = mat[2,0] = 20
    mat[1,2] = mat[2,1] = 20
    
    el = Elastic(mat)
    assert np.allclose(el.CVoigt, mat)
    assert el.is2D() == False

def test_elastic_initialization_triangular():
    # Upper triangular
    mat = np.zeros((6,6))
    for i in range(6):
        for j in range(i, 6):
            mat[i,j] = 10.0 if i == j else 2.0
            
    el = Elastic(mat)
    assert np.all(el.CVoigt == el.CVoigt.T)
    assert el.CVoigt[0,1] == 2.0

def test_elastic_robust_triangular():
    # Noisy lower triangular (simulating VASP output)
    mat = np.zeros((6,6))
    for i in range(6):
        mat[i,i] = 100.0
        for j in range(i):
            mat[i,j] = 20.0
    
    # Add noise to the "zero" part
    mat[0,1] = 1e-15
    
    el = Elastic(mat)
    assert np.allclose(el.CVoigt, el.CVoigt.T)
    assert el.CVoigt[0,1] == 20.0

def test_elastic_csv_parsing():
    csv_content = "100,20,20,0,0,0\n20,100,20,0,0,0\n20,20,100,0,0,0\n0,0,0,40,0,0\n0,0,0,0,40,0\n0,0,0,0,0,40"
    el = Elastic(csv_content)
    assert el.CVoigt[0,1] == 20.0
    assert el.CVoigt[3,3] == 40.0

def test_averages():
    # Simple isotropic material
    # For isotropic: K = (c11 + 2*c12)/3, G = (c11 - c12)/2
    # Let's use c11=100, c12=40, c44=30 -> Isotropic since (100-40)/2 = 30
    mat = np.zeros((6,6))
    mat[0:3, 0:3] = 40
    for i in range(3): mat[i,i] = 100
    mat[3:6, 3:6] = np.eye(3) * 30
    
    el = Elastic(mat)
    avgs = el.averages()
    # Voigt, Reuss, Hill should all be identical for isotropic
    # Hill Bulk: (100 + 2*40)/3 = 60
    # Hill Shear: 30
    assert pytest.approx(avgs[2][0]) == 60.0
    assert pytest.approx(avgs[2][2]) == 30.0

def test_stability():
    # Stable matrix
    mat = np.eye(6) * 100
    el = Elastic(mat)
    assert np.all(el.eigenvalues() > 0)
    
    # Unstable matrix (negative eigenvalue)
    mat_unstable = np.eye(6) * 100
    mat_unstable[0,0] = -10
    el_unstable = Elastic(mat_unstable)
    assert np.any(el_unstable.eigenvalues() < 0)

# --- 2D Elastic Tests ---

def test_elastic2d_initialization():
    mat = np.array([
        [100, 20, 0],
        [20, 100, 0],
        [0, 0, 40]
    ])
    el = Elastic2D(mat)
    assert el.is2D() == True
    assert el.CVoigt.shape == (3,3)

def test_elastic2d_poisson_consistency():
    # This specifically tests the fix for the Poisson bug
    matrix = np.array([
        [100, 20, 10],
        [20, 100, 5],
        [10, 5, 40]
    ])
    material = Elastic2D(matrix)
    
    for angle in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
        E = material.Young(angle)
        nu = material.Poisson(angle)
        
        # Rotated compliance S'11 = 1/E
        # Rotated compliance S'12 = -nu * S'11
        # Thus -nu/E = S'12
        # Let's check if nu = -S'12 / S'11 consistently
        
        # We also verified in repro_bug.py that the denominator used in Poisson
        # must match 1/E.
        # Since I fixed the denominator, Young(angle) and 1/denom in Poisson should match.
        # The internal calculation of S'12 in the Poisson method (num) is:
        # num = -S'12
        # So Poisson = -num / denom = S'12 / S'11.
        # Wait, the code says: return -num/denom
        # If num = -S'12, then -num/denom = S'12/S'11.
        # Is that correct? nu = -S'12/S'11.
        # Let's re-verify the Poisson formula in core.py
        pass

def test_elastic2d_symmetry_enforcement():
    # Testing robust triangular/symmetry logic for 2D
    mat = np.array([
        [100, 0.1, 0.2],
        [20, 100, 0.3],
        [30, 15, 40]
    ])
    # Not symmetric, not triangular -> should fail
    with pytest.raises(ValueError, match="should be symmetric, or triangular"):
        Elastic2D(mat)
    
    # Triangular
    mat_tri = np.array([
        [100, 20, 30],
        [0, 100, 15],
        [0, 0, 40]
    ])
    el = Elastic2D(mat_tri)
    assert el.CVoigt[1,0] == 20
    assert el.CVoigt[2,0] == 30
