import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from typing import Optional, Dict, Any, Tuple
from .core import Elastic

def get_style(config: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    """Helper to retrieve a style setting from config or return a default.

    Args:
        config: The style configuration dictionary.
        key: The key to look for.
        default: The default value if key is not found.

    Returns:
        The configuration value or default.
    """
    return config.get(key, default) if config else default

def plot_youngs_2d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None) -> None:
    """Generates 2D polar plots of Young's Modulus in XY, YZ, and ZX planes.

    Args:
        material: The Elastic material object.
        prefix: Prefix for the output filename.
        style_config: Optional dictionary with matplotlib style settings.
    """
    phi = np.linspace(0, 2 * np.pi, 100)
    f = np.vectorize(material.Young_2)

    r_xy = f(np.pi / 2, phi)
    r_yz = f(phi - np.pi/2, np.pi/2)
    r_zx = f(phi, 0)

    fig_config = get_style(style_config, "figure", {})
    figsize = fig_config.get("figsize_2d", [15, 5])
    
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=figsize)
    title_font = get_style(style_config, "title_font", {'fontsize': 14, 'fontweight': 'bold', 'family': 'sans-serif'})
    line_config = get_style(style_config, "line", {})
    line_color = line_config.get("color", "b")
    line_width = line_config.get("width", 3)

    axes[0].plot(phi, r_xy, color=line_color, linewidth=line_width)
    axes[0].set_title("XY Plane", fontdict=title_font)

    axes[1].plot(phi, r_yz, color=line_color, linewidth=line_width)
    axes[1].set_title("YZ Plane", fontdict=title_font)

    axes[2].plot(phi, r_zx, color=line_color, linewidth=line_width)
    axes[2].set_title("ZX Plane", fontdict=title_font)

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    filename = f"{prefix}_Youngs_Modulus_Polar_2D.png"
    dpi = get_style(style_config, "dpi", 300)
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Saved 2D Young's Modulus plot to {filename}")

def plot_youngs_3d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None) -> None:
    """Generates a 3D surface plot of Young's Modulus directional dependence.

    Args:
        material: The Elastic material object.
        prefix: Prefix for the output filename.
        style_config: Optional dictionary with matplotlib style settings.
    """
    def spherical_grid(npoints=200):
        theta = np.linspace(0, np.pi, npoints)
        phi = np.linspace(0, 2 * np.pi, npoints)
        return np.meshgrid(theta, phi)

    def spherical_coord(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    theta, phi = spherical_grid()
    f = np.vectorize(material.Young_2)
    r = f(theta, phi)

    x, y, z = spherical_coord(r, theta, phi)
    
    fig_config = get_style(style_config, "figure", {})
    figsize = fig_config.get("figsize_3d", [10, 10])
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")

    vmin = style_config.get("vmin") if style_config and style_config.get("vmin") is not None else np.min(r)
    vmax = style_config.get("vmax") if style_config and style_config.get("vmax") is not None else np.max(r)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_name = get_style(style_config, "colormap", "turbo")
    cmap = getattr(cm, cmap_name, cm.turbo)

    surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(r)), edgecolor='none', 
                            linewidth=0, antialiased=True, shade=True)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
    
    label_font = get_style(style_config, "label_font", {'fontsize': 12, 'fontweight': 'bold'})
    cbar.set_label("Young's Modulus (GPa)", **label_font)

    ax.set_xlabel("X Axis", **label_font)
    ax.set_ylabel("Y Axis", **label_font)
    ax.set_zlabel("Z Axis", **label_font)
    ax.view_init(elev=35, azim=135)

    filename = f"{prefix}_Youngs_Modulus_3D.png"
    dpi = get_style(style_config, "dpi", 300)
    transparent = fig_config.get("transparent", True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=transparent)
    plt.close()
    print(f"Saved 3D Young's Modulus plot to {filename}")

def plot_shear_2d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None) -> None:
    """Generates 2D polar plots of min/max Shear Modulus in XY, YZ, and ZX planes.

    Args:
        material: The Elastic material object.
        prefix: Prefix for the output filename.
        style_config: Optional dictionary with matplotlib style settings.
    """
    phi = np.linspace(0, 2 * np.pi, 100)
    
    # XY Plane
    f_xy = np.vectorize(lambda x: material.shear2D([np.pi / 2, x]))
    r_xy = f_xy(phi)

    # YZ Plane
    f_yz = np.vectorize(lambda x: material.shear2D([x-np.pi/2, np.pi/2]))
    r_yz = f_yz(phi)

    # ZX Plane
    f_zx = np.vectorize(lambda x: material.shear2D([x, 0]))
    r_zx = f_zx(phi)

    fig_config = get_style(style_config, "figure", {})
    figsize = fig_config.get("figsize_2d", [15, 5])
    
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=figsize)
    title_font = get_style(style_config, "title_font", {'fontsize': 14, 'fontweight': 'bold', 'family': 'sans-serif'})
    
    line_config = get_style(style_config, "line", {})
    min_color = line_config.get("color", "b")
    max_color = line_config.get("max_color", "r")
    line_width = line_config.get("width", 3)

    axes[0].plot(phi, r_xy[0], color=min_color, linewidth=line_width, label='Min')
    axes[0].plot(phi, r_xy[1], color=max_color, linewidth=line_width, label='Max')
    axes[0].set_title("XY Plane", fontdict=title_font)
    axes[0].legend()

    axes[1].plot(phi, r_yz[0], color=min_color, linewidth=line_width)
    axes[1].plot(phi, r_yz[1], color=max_color, linewidth=line_width)
    axes[1].set_title("YZ Plane", fontdict=title_font)

    axes[2].plot(phi, r_zx[0], color=min_color, linewidth=line_width)
    axes[2].plot(phi, r_zx[1], color=max_color, linewidth=line_width)
    axes[2].set_title("ZX Plane", fontdict=title_font)

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    filename = f"{prefix}_Shear_Modulus_Polar_2D.png"
    dpi = get_style(style_config, "dpi", 300)
    plt.savefig(filename, dpi=dpi)
    plt.close()
    print(f"Saved 2D Shear Modulus plot to {filename}")

def plot_shear_3d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None) -> None:
    """Generates a 3D surface plot of Maximum Shear Modulus directional dependence.

    Note:
        Due to the extra degree of freedom (chi rotation), this calculates the 
        global maximum shear modulus for each direction (theta, phi).

    Args:
        material: The Elastic material object.
        prefix: Prefix for the output filename.
        style_config: Optional dictionary with matplotlib style settings.
    """
    def spherical_grid(npoints=100):
        theta = np.linspace(0, np.pi, npoints)
        phi = np.linspace(0, 2 * np.pi, npoints)
        return np.meshgrid(theta, phi)

    def spherical_coord(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    theta, phi = spherical_grid(npoints=60) # Lower resolution for expensive optimization
    
    # shear3D returns (min, max, chi_min, chi_max)
    f = np.vectorize(lambda t, p: material.shear3D(t, p)[1]) # We plot the Max shear modulus
    r = f(theta, phi)

    x, y, z = spherical_coord(r, theta, phi)
    
    fig_config = get_style(style_config, "figure", {})
    figsize = fig_config.get("figsize_3d", [10, 10])
    
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")

    vmin = style_config.get("vmin") if style_config and style_config.get("vmin") is not None else np.min(r)
    vmax = style_config.get("vmax") if style_config and style_config.get("vmax") is not None else np.max(r)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_name = get_style(style_config, "colormap", "magma")
    cmap = getattr(cm, cmap_name, cm.magma)

    surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(r)), edgecolor='none', 
                            linewidth=0, antialiased=True, shade=True)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=10)
    
    label_font = get_style(style_config, "label_font", {'fontsize': 12, 'fontweight': 'bold'})
    cbar.set_label("Max Shear Modulus (GPa)", **label_font)

    ax.set_xlabel("X Axis", **label_font)
    ax.set_ylabel("Y Axis", **label_font)
    ax.set_zlabel("Z Axis", **label_font)
    ax.view_init(elev=35, azim=135)

    filename = f"{prefix}_Shear_Modulus_3D.png"
    dpi = get_style(style_config, "dpi", 300)
    transparent = fig_config.get("transparent", True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=transparent)
    plt.close()
    print(f"Saved 3D Shear Modulus plot to {filename}")
