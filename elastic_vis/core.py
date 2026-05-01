# -*- coding: utf-8 -*-

import json
import math
from typing import List, Tuple, Union, Optional, Any

import numpy as np
from scipy import optimize


def dirVec(theta: float, phi: float) -> List[float]:
    """Return a unit vector associated with angles theta and phi.

    Args:
        theta: The polar angle in radians.
        phi: The azimuthal angle in radians.

    Returns:
        A list of three floats representing the unit vector [x, y, z].
    """
    return [math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)]


def dirVec1(theta: float, phi: float, chi: float) -> List[float]:
    """Return the first unit vector associated with angles (theta, phi, chi).

    Args:
        theta: The polar angle in radians.
        phi: The azimuthal angle in radians.
        chi: The rotation angle in radians.

    Returns:
        A list of three floats representing the first unit vector.
    """
    return [math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)]


def dirVec2(theta: float, phi: float, chi: float) -> List[float]:
    """Return the second unit vector associated with angles (theta, phi, chi).

    Args:
        theta: The polar angle in radians.
        phi: The azimuthal angle in radians.
        chi: The rotation angle in radians.

    Returns:
        A list of three floats representing the second unit vector.
    """
    return [math.cos(theta)*math.cos(phi)*math.cos(chi) - math.sin(phi)*math.sin(chi),
            math.cos(theta)*math.sin(phi)*math.cos(chi) + math.cos(phi)*math.sin(chi),
            - math.sin(theta)*math.cos(chi)]


def minimize(func: Any, dim: int) -> Tuple[np.ndarray, float]:
    """Find the global minimum of a function with 2 or 3 parameters.

    Args:
        func: The function to minimize.
        dim: Dimensionality of the function (2 for theta, phi; 3 for theta, phi, chi).

    Returns:
        A tuple (optimal_parameters, minimum_value).
    """
    if dim == 2:
        r = ((0, np.pi), (0, np.pi))
        n = 25
    elif dim == 3:
        r = ((0, np.pi), (0, np.pi), (0, np.pi))
        n = 10
    else:
        raise ValueError("dim must be 2 or 3")

    return optimize.brute(func, r, Ns=n, full_output=True, finish=optimize.fmin)[0:2]


def maximize(func: Any, dim: int) -> Tuple[np.ndarray, float]:
    """Find the global maximum of a function with 2 or 3 parameters.

    Args:
        func: The function to maximize.
        dim: Dimensionality of the function (2 or 3).

    Returns:
        A tuple (optimal_parameters, maximum_value).
    """
    res = minimize(lambda x: -func(x), dim)
    return (res[0], -res[1])


class Elastic:
    """An elastic tensor, along with methods to access its properties.

    Attributes:
        CVoigt (np.ndarray): The 6x6 stiffness matrix in Voigt notation.
        SVoigt (np.ndarray): The 6x6 compliance matrix in Voigt notation.
        Smat (List): The 4th order compliance tensor in Cartesian coordinates.
    """

    def __init__(self, s: Union[str, list, np.ndarray]):
        """Initialize the elastic tensor.

        Args:
            s: Input stiffness data. Can be a 6-line string, a list of lists,
               a numpy array, or a JSON string representation.

        Raises:
            ValueError: If the input matrix is not 6x6, is invalid, non-square,
                or non-symmetric.
            TypeError: If the input corresponds to a 2D material.
        """
        if isinstance(s, str):
            try:
                s = json.loads(s)
            except:
                pass

        if isinstance(s, (np.ndarray, list)):
            mat = s
        elif isinstance(s, str):
            s = s.replace("|", " ").replace("(", " ").replace(")", " ")
            lines = [line for line in s.split('\n') if line.strip()]
            if len(lines) == 3:
                raise TypeError("is a 2D material")
            if len(lines) != 6:
                raise ValueError("should have three or six rows")
            try:
                mat = [list(map(float, line.split())) for line in lines]
            except:
                raise ValueError("not all entries are numbers")
        else:
            raise ValueError("invalid argument as matrix")

        try:
            mat = np.array(mat)
        except:
            raise ValueError("input must be a 6x6 stiffness matrix")

        if not isinstance(mat, np.ndarray):
            raise ValueError("input must be a 6x6 stiffness matrix")
        if mat.shape == (3,3):
            raise TypeError("is a 2D material")
        if mat.shape != (6,6):
            raise ValueError("input must be a 6x6 stiffness matrix")

        if np.linalg.norm(np.tril(mat, -1)) == 0:
            mat = mat + np.triu(mat, 1).transpose()
        if np.linalg.norm(np.triu(mat, 1)) == 0:
            mat = mat + np.tril(mat, -1).transpose()
        
        if np.linalg.norm(mat - mat.transpose()) > 1e-3:
            raise ValueError("should be symmetric, or triangular")
        elif np.linalg.norm(mat - mat.transpose()) > 0:
            mat = 0.5 * (mat + mat.transpose())

        self.CVoigt = mat
        try:
            self.SVoigt = np.linalg.inv(self.CVoigt)
        except:
            raise ValueError("matrix is singular")

        VoigtMat = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        def SVoigtCoeff(p, q):
            return 1. / ((1+p//3)*(1+q//3))

        self.Smat = [[[[ SVoigtCoeff(VoigtMat[i][j], VoigtMat[k][l]) * self.SVoigt[VoigtMat[i][j]][VoigtMat[k][l]]
                         for i in range(3) ] for j in range(3) ] for k in range(3) ] for l in range(3) ]

    def is2D(self) -> bool:
        """Returns True if the material is 2D."""
        return False

    def isOrthorhombic(self) -> bool:
        """Checks if the material is orthorhombic."""
        def iszero(x): return (abs(x) < 1.e-3)
        return (iszero(self.CVoigt[0][3]) and iszero(self.CVoigt[0][4]) and iszero(self.CVoigt[0][5])
                and iszero(self.CVoigt[1][3]) and iszero(self.CVoigt[1][4]) and iszero(self.CVoigt[1][5])
                and iszero(self.CVoigt[2][3]) and iszero(self.CVoigt[2][4]) and iszero(self.CVoigt[2][5])
                and iszero(self.CVoigt[3][4]) and iszero(self.CVoigt[3][5]) and iszero(self.CVoigt[4][5]))

    def isCubic(self) -> bool:
        """Checks if the material is cubic."""
        def iszero(x): return (abs(x) < 1.e-3)
        return (iszero(self.CVoigt[0][3]) and iszero(self.CVoigt[0][4]) and iszero(self.CVoigt[0][5])
                and iszero(self.CVoigt[1][3]) and iszero(self.CVoigt[1][4]) and iszero(self.CVoigt[1][5])
                and iszero(self.CVoigt[2][3]) and iszero(self.CVoigt[2][4]) and iszero(self.CVoigt[2][5])
                and iszero(self.CVoigt[3][4]) and iszero(self.CVoigt[3][5]) and iszero(self.CVoigt[4][5])
                and iszero(self.CVoigt[0][0] - self.CVoigt[1][1]) and iszero(self.CVoigt[0][0] - self.CVoigt[2][2])
                and iszero(self.CVoigt[3][3] - self.CVoigt[4][4]) and iszero(self.CVoigt[3][3] - self.CVoigt[5][5])
                and iszero(self.CVoigt[0][1] - self.CVoigt[0][2]) and iszero(self.CVoigt[0][1] - self.CVoigt[1][2]))

    def Young(self, x: Union[List[float], np.ndarray]) -> float:
        """Calculates Young's Modulus for a given direction.

        Args:
            x: A list/array [theta, phi] defining the direction.

        Returns:
            The Young's Modulus in GPa.
        """
        a = dirVec(x[0], x[1])
        r = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/r

    def Young_2(self, theta: float, phi: float) -> float:
        """Calculates Young's Modulus for a given theta and phi.

        Args:
            theta: Polar angle in radians.
            phi: Azimuthal angle in radians.

        Returns:
            The Young's Modulus in GPa.
        """
        return self.Young([theta, phi])

    def LC(self, x: Union[List[float], np.ndarray]) -> float:
        """Calculates Linear Compressibility for a given direction.

        Args:
            x: A list/array [theta, phi] defining the direction.

        Returns:
            The linear compressibility.
        """
        a = dirVec(x[0], x[1])
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        return 1000 * r

    def LC_2(self, theta: float, phi: float) -> float:
        """Calculates Linear Compressibility for a given theta and phi."""
        return self.LC([theta, phi])

    def shear(self, x: Union[List[float], np.ndarray]) -> float:
        """Calculates Shear Modulus for a given direction and orientation.

        Args:
            x: A list/array [theta, phi, chi] defining direction and orientation.

        Returns:
            The Shear Modulus in GPa.
        """
        a = dirVec(x[0], x[1])
        b = dirVec2(x[0], x[1], x[2])
        r = sum([ a[i]*b[j]*a[k]*b[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/(4*r)

    def Poisson(self, x: Union[List[float], np.ndarray]) -> float:
        """Calculates Poisson's Ratio for a given direction and orientation.

        Args:
            x: A list/array [theta, phi, chi] defining direction and orientation.

        Returns:
            The Poisson's Ratio.
        """
        a = dirVec(x[0], x[1])
        b = dirVec2(x[0], x[1], x[2])
        r1 = sum([ a[i]*a[j]*b[k]*b[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        r2 = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return -r1/r2

    def averages(self) -> List[List[float]]:
        """Calculates Voigt, Reuss, and Hill averages.

        Returns:
            A list containing [KV, EV, GV, nV], [KR, ER, GR, nR], [KH, EH, GH, nH]
            where K is Bulk modulus, E is Young's modulus, G is Shear modulus,
            and n is Poisson's ratio.
        """
        A = (self.CVoigt[0][0] + self.CVoigt[1][1] + self.CVoigt[2][2]) / 3
        B = (self.CVoigt[1][2] + self.CVoigt[0][2] + self.CVoigt[0][1]) / 3
        C = (self.CVoigt[3][3] + self.CVoigt[4][4] + self.CVoigt[5][5]) / 3
        a = (self.SVoigt[0][0] + self.SVoigt[1][1] + self.SVoigt[2][2]) / 3
        b = (self.SVoigt[1][2] + self.SVoigt[0][2] + self.SVoigt[0][1]) / 3
        c = (self.SVoigt[3][3] + self.SVoigt[4][4] + self.SVoigt[5][5]) / 3

        KV = (A + 2*B) / 3
        GV = (A - B + 3*C) / 5

        KR = 1 / (3*a + 6*b)
        GR = 5 / (4*a - 4*b + 3*c)

        KH = (KV + KR) / 2
        GH = (GV + GR) / 2

        return [ [KV, 1/(1/(3*GV) + 1/(9*KV)), GV, (1 - 3*GV/(3*KV+GV))/2],
                 [KR, 1/(1/(3*GR) + 1/(9*KR)), GR, (1 - 3*GR/(3*KR+GR))/2],
                 [KH, 1/(1/(3*GH) + 1/(9*KH)), GH, (1 - 3*GH/(3*KH+GH))/2] ]

    def eigenvalues(self) -> np.ndarray:
        """Calculates the eigenvalues of the stiffness matrix."""
        return np.sort(np.linalg.eigvalsh(self.CVoigt))

    def shear2D(self, x: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        """Finds min and max shear modulus for a direction [theta, phi]."""
        def func(chi): return self.shear([x[0], x[1], chi])
        
        # Use a robust 1D bounded optimizer
        res_min = optimize.minimize_scalar(func, bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        res_max = optimize.minimize_scalar(lambda c: -func(c), bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        
        return (float(res_min.fun), -float(res_max.fun))

    def shear3D(self, theta: float, phi: float, guess1: Optional[float] = None, guess2: Optional[float] = None) -> Tuple[float, float, float, float]:
        """Finds min and max shear modulus for a direction (theta, phi)."""
        def func(chi): return self.shear([theta, phi, chi])
        
        res_min = optimize.minimize_scalar(func, bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        res_max = optimize.minimize_scalar(lambda c: -func(c), bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        
        return (float(res_min.fun), -float(res_max.fun), float(res_min.x), float(res_max.x))

    def Poisson2D(self, x: Union[List[float], np.ndarray]) -> Tuple[float, float]:
        """Finds min and max Poisson's ratio for a direction [theta, phi]."""
        def func(chi): return self.Poisson([x[0], x[1], chi])
        
        res_min = optimize.minimize_scalar(func, bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        res_max = optimize.minimize_scalar(lambda c: -func(c), bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        
        return (float(res_min.fun), -float(res_max.fun))

    def Poisson3D(self, theta: float, phi: float) -> Tuple[float, float, float, float]:
        """Finds Poisson's ratio extremes for a direction (theta, phi)."""
        def func(chi): return self.Poisson([theta, phi, chi])
        
        res_min = optimize.minimize_scalar(func, bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        res_max = optimize.minimize_scalar(lambda c: -func(c), bounds=(0, np.pi), method='bounded', options={'xatol': 1e-5})
        
        return (float(res_min.fun), -float(res_max.fun), float(res_min.x), float(res_max.x))


class ElasticOrtho(Elastic):
    """An elastic tensor for orthorhombic systems with optimized calculations."""

    def __init__(self, arg: Union[str, Elastic]):
        """Initialize from a matrix string or an existing Elastic object."""
        if isinstance(arg, str):
            super().__init__(arg)
        elif isinstance(arg, Elastic):
            self.CVoigt = arg.CVoigt
            self.SVoigt = arg.SVoigt
            self.Smat = arg.Smat
        else:
            raise TypeError("Argument should be string or Elastic object")

    def Young(self, x: Union[List[float], np.ndarray]) -> float:
        """Optimized Young's Modulus calculation for orthorhombic symmetry."""
        ct2 = math.cos(x[0])**2
        st2 = 1 - ct2
        cf2 = math.cos(x[1])**2
        sf2 = 1 - cf2
        s11, s22, s33 = self.Smat[0][0][0][0], self.Smat[1][1][1][1], self.Smat[2][2][2][2]
        s44, s55, s66 = 4*self.Smat[1][2][1][2], 4*self.Smat[0][2][0][2], 4*self.Smat[0][1][0][1]
        s12, s13, s23 = self.Smat[0][0][1][1], self.Smat[0][0][2][2], self.Smat[1][1][2][2]
        return 1/(ct2**2*s33 + 2*cf2*ct2*s13*st2 + cf2*ct2*s55*st2 + 2*ct2*s23*sf2*st2 + ct2*s44*sf2*st2 + cf2**2*s11*st2**2 + 2*cf2*s12*sf2*st2**2 + cf2*s66*sf2*st2**2 + s22*sf2**2*st2**2)

    def LC(self, x: Union[List[float], np.ndarray]) -> float:
        """Optimized Linear Compressibility calculation for orthorhombic symmetry."""
        ct2, cf2 = math.cos(x[0])**2, math.cos(x[1])**2
        s11, s22, s33 = self.Smat[0][0][0][0], self.Smat[1][1][1][1], self.Smat[2][2][2][2]
        s12, s13, s23 = self.Smat[0][0][1][1], self.Smat[0][0][2][2], self.Smat[1][1][2][2]
        return 1000 * (ct2 * (s13 + s23 + s33) + (cf2 * (s11 + s12 + s13) + (s12 + s22 + s23) * (1 - cf2)) * (1 - ct2))


class Elastic2D:
    """Elastic tensor for 2D materials."""

    def __init__(self, s: Union[str, list, np.ndarray]):
        """Initialize the 2D elastic tensor."""
        if isinstance(s, str):
            try:
                if isinstance(json.loads(s), list):
                    s = json.loads(s)
            except:
                pass

        if isinstance(s, str):
            s = s.replace("|", " ").replace("(", " ").replace(")", " ")
            lines = [line for line in s.split('\n') if line.strip()]
            if len(lines) != 3:
                raise ValueError("should have three rows")
            try:
                mat = [list(map(float, line.split())) for line in lines]
            except:
                raise ValueError("not all entries are numbers")
        elif isinstance(s, (list, np.ndarray)):
            mat = s
        else:
            raise ValueError("invalid argument as matrix")

        try:
            mat = np.array(mat)
        except:
            raise ValueError("input must be a 3x3 stiffness matrix")

        if not isinstance(mat, np.ndarray) or mat.shape != (3,3):
            raise ValueError("input must be a 3x3 stiffness matrix")

        if np.linalg.norm(np.tril(mat, -1)) == 0:
            mat = mat + np.triu(mat, 1).transpose()
        if np.linalg.norm(np.triu(mat, 1)) == 0:
            mat = mat + np.tril(mat, -1).transpose()
        if np.linalg.norm(mat - mat.transpose()) > 1e-3:
            raise ValueError("should be symmetric, or triangular")
        elif np.linalg.norm(mat - mat.transpose()) > 0:
            mat = 0.5 * (mat + mat.transpose())

        self.CVoigt = mat
        try:
            self.SVoigt = np.linalg.inv(self.CVoigt)
        except:
            raise ValueError("matrix is singular")

        self.s11, self.s12, self.s16 = self.SVoigt[0][0], self.SVoigt[0][1], self.SVoigt[0][2]
        self.s22, self.s26, self.s66 = self.SVoigt[1][1], self.SVoigt[1][2], self.SVoigt[2][2]

    def is2D(self) -> bool:
        """Returns True."""
        return True

    def Young(self, theta: float) -> float:
        """Calculates 2D Young's Modulus for a given angle."""
        ct, st = math.cos(theta), math.sin(theta)
        return 1/(self.s11*ct**4 + self.s22*st**4 + 2*self.s16*ct**3*st + 2*self.s26*ct*st**3 + (2*self.s12+self.s66)*ct**2*st**2)

    def shear(self, theta: float) -> float:
        """Calculates 2D Shear Modulus for a given angle."""
        ct, st = math.cos(theta), math.sin(theta)
        calc = ((self.s11 + self.s22 - 2*self.s12)*ct**2*st**2
                + self.s66/4*(ct**4 + st**4 - 2*st**2*ct**2)
                + self.s16*(st**3*ct - ct**3*st)
                + self.s26*(ct**3*st - st**3*ct))
        return 1 / (4 * calc)

    def Poisson(self, theta: float) -> float:
        """Calculates 2D Poisson's Ratio for a given angle."""
        ct, st = math.cos(theta), math.sin(theta)
        num = ((self.s11 + self.s22 - self.s66)*ct**2*st**2
               + self.s12*(ct**4 + st**4)
               + self.s16*(st**3*ct - ct**3*st)
               + self.s26*(ct**3*st - st**3*ct))
        denom = ((2*self.s12 + self.s66)*ct**2*st**2
                 + self.s11*ct**4 + self.s22*st**4
                 + 2*self.s16*st**3*ct
                 + 2*self.s26*ct**3*st)
        return -num/denom

    def eigenvalues(self) -> np.ndarray:
        """Calculates eigenvalues of the 2D stiffness matrix."""
        return np.sort(np.linalg.eigvalsh(self.CVoigt))
