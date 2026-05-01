import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from typing import Optional, Dict, Any, Tuple
from .core import Elastic

def set_publishable_style(style_config: Optional[Dict[str, Any]] = None):
    """Sets a clean, publication-quality style for matplotlib plots."""
    # Base defaults
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 1.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "grid.linestyle": "--",
        "savefig.bbox": "tight"
    })
    
    if style_config:
        gen = style_config.get("general", {})
        if "font_family" in gen:
            plt.rcParams["font.family"] = gen["font_family"]
        
        # Apply global sizes if defined at root (backward compatibility)
        if "label_font" in style_config:
            fs = style_config["label_font"].get("fontsize", 14)
            plt.rcParams["axes.labelsize"] = fs
            plt.rcParams["xtick.labelsize"] = fs - 2
            plt.rcParams["ytick.labelsize"] = fs - 2

def get_style(config: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    """Helper to retrieve a style setting from config or return a default."""
    return config.get(key, default) if config else default

def build_filename(prefix: str, suffix: str, output_format: str) -> str:
    """Build a plot filename with the requested output format."""
    return f"{prefix}_{suffix}.{output_format}"

def plot_youngs_2d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None, show: bool = False, output_format: str = "png") -> None:
    """Generates 2D polar plots of Young's Modulus."""
    set_publishable_style(style_config)
    
    cfg_2d = get_style(style_config, "plots_2d", {})
    phi = np.linspace(0, 2 * np.pi, 500)
    f = np.vectorize(material.Young_2)

    r_xy = f(np.pi / 2, phi)
    r_yz = f(phi - np.pi/2, np.pi/2)
    r_zx = f(phi, 0)

    figsize = cfg_2d.get("figsize", [15, 6])
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=figsize)
    
    title_font = {
        'fontsize': cfg_2d.get("title_size", 18),
        'fontweight': cfg_2d.get("font_weight", "bold"),
        'family': cfg_2d.get("font_family", plt.rcParams["font.family"])
    }
    
    label_font_size = cfg_2d.get("label_size", 12)
    line_color = cfg_2d.get("line_color", "#0072B2")
    line_width = cfg_2d.get("line_width", 3.0)

    planes = ["XY Plane", "YZ Plane", "ZX Plane"]
    data = [r_xy, r_yz, r_zx]

    for i, ax in enumerate(axes):
        ax.plot(phi, data[i], color=line_color, linewidth=line_width)
        ax.set_title(planes[i], pad=20, fontdict=title_font)
        ax.tick_params(labelsize=label_font_size)
        ax.grid(True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

    plt.tight_layout()
    filename = build_filename(prefix, "Youngs_Modulus_Polar_2D", output_format)
    dpi = get_style(style_config.get("general", {}), "dpi", 300) if style_config else 300
    plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    print(f"Saved 2D Young's Modulus plot to {filename}")

def plot_youngs_3d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None, show: bool = False, output_format: str = "png") -> None:
    """Generates a 3D surface plot of Young's Modulus."""
    set_publishable_style(style_config)
    
    cfg_3d = get_style(style_config, "plots_3d", {})
    
    def spherical_grid(npoints=300):
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
    
    figsize = cfg_3d.get("figsize", [12, 10])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1,1,1])
    
    vmin = cfg_3d.get("vmin") if cfg_3d.get("vmin") is not None else np.min(r)
    vmax = cfg_3d.get("vmax") if cfg_3d.get("vmax") is not None else np.max(r)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_name = cfg_3d.get("colormap", "plasma")
    cmap = getattr(cm, cmap_name, cm.plasma)

    rcount = cfg_3d.get("rcount", 200)
    ccount = cfg_3d.get("ccount", 200)

    surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(r)), edgecolor='none', 
                            rcount=rcount, ccount=ccount, antialiased=True, shade=True)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    
    label_font = {
        'fontsize': cfg_3d.get("label_size", 14),
        'fontweight': cfg_3d.get("font_weight", "normal"),
        'family': cfg_3d.get("font_family", plt.rcParams["font.family"])
    }
    cbar.set_label("Young's Modulus (GPa)", labelpad=15, **label_font)

    max_val = np.max([np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))])
    limit = max_val * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel("X (GPa)", labelpad=15, **label_font)
    ax.set_ylabel("Y (GPa)", labelpad=15, **label_font)
    ax.set_zlabel("Z (GPa)", labelpad=15, **label_font)
    
    ax.tick_params(axis='both', which='major', labelsize=label_font['fontsize']-2)
    
    elev = cfg_3d.get("view_elevation", 20)
    azim = cfg_3d.get("view_azimuth", 45)
    ax.view_init(elev=elev, azim=azim)

    filename = build_filename(prefix, "Youngs_Modulus_3D", output_format)
    gen = style_config.get("general", {}) if style_config else {}
    dpi = gen.get("dpi", 300)
    transparent = gen.get("transparent", True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.8, transparent=transparent)
    if show:
        plt.show()
    plt.close()
    print(f"Saved 3D Young's Modulus plot to {filename}")

def plot_shear_2d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None, show: bool = False, output_format: str = "png") -> None:
    """Generates 2D polar plots of Shear Modulus."""
    set_publishable_style(style_config)
    
    cfg_2d = get_style(style_config, "plots_2d", {})
    phi = np.linspace(0, 2 * np.pi, 500)
    
    f_xy = np.vectorize(lambda x: material.shear2D([np.pi / 2, x]))
    r_xy_min, r_xy_max = f_xy(phi)

    f_yz = np.vectorize(lambda x: material.shear2D([x - np.pi/2, np.pi/2]))
    r_yz_min, r_yz_max = f_yz(phi)

    f_zx = np.vectorize(lambda x: material.shear2D([x, 0]))
    r_zx_min, r_zx_max = f_zx(phi)

    figsize = cfg_2d.get("figsize", [15, 6])
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=figsize)
    
    title_font = {
        'fontsize': cfg_2d.get("title_size", 18),
        'fontweight': cfg_2d.get("font_weight", "bold"),
        'family': cfg_2d.get("font_family", plt.rcParams["font.family"])
    }
    
    label_font_size = cfg_2d.get("label_size", 12)
    min_color = cfg_2d.get("min_color", "#0072B2")
    max_color = cfg_2d.get("max_color", "#D55E00")
    line_width = cfg_2d.get("line_width", 3.0)

    planes = ["XY Plane", "YZ Plane", "ZX Plane"]
    mins = [r_xy_min, r_yz_min, r_zx_min]
    maxs = [r_xy_max, r_yz_max, r_zx_max]

    for i, ax in enumerate(axes):
        ax.plot(phi, mins[i], color=min_color, linewidth=line_width, label='Min')
        ax.plot(phi, maxs[i], color=max_color, linewidth=line_width, label='Max')
        ax.set_title(planes[i], pad=25, fontdict=title_font)
        ax.tick_params(labelsize=label_font_size)
        ax.grid(True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        if i == 2:
            ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0.1), fontsize=label_font_size)

    plt.tight_layout()
    filename = build_filename(prefix, "Shear_Modulus_Polar_2D", output_format)
    dpi = get_style(style_config.get("general", {}), "dpi", 300) if style_config else 300
    plt.savefig(filename, dpi=dpi)
    if show:
        plt.show()
    plt.close()
    print(f"Saved 2D Shear Modulus plot to {filename}")

def plot_shear_3d(material: Elastic, prefix: str, style_config: Optional[Dict[str, Any]] = None, show: bool = False, output_format: str = "png") -> None:
    """Generates a 3D surface plot of Maximum Shear Modulus."""
    set_publishable_style(style_config)
    
    cfg_3d = get_style(style_config, "plots_3d", {})
    
    def spherical_grid(npoints=200):
        theta = np.linspace(0, np.pi, npoints)
        phi = np.linspace(0, 2 * np.pi, npoints)
        return np.meshgrid(theta, phi)

    def spherical_coord(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    theta, phi = spherical_grid(npoints=150)
    f = np.vectorize(lambda t, p: material.shear3D(t, p)[1])
    r = f(theta, phi)

    x, y, z = spherical_coord(r, theta, phi)
    
    figsize = cfg_3d.get("figsize", [12, 10])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1,1,1])

    vmin = cfg_3d.get("vmin") if cfg_3d.get("vmin") is not None else np.min(r)
    vmax = cfg_3d.get("vmax") if cfg_3d.get("vmax") is not None else np.max(r)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_name = cfg_3d.get("colormap", "magma")
    cmap = getattr(cm, cmap_name, cm.magma)

    rcount = cfg_3d.get("rcount", 150)
    ccount = cfg_3d.get("ccount", 150)

    surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(r)), edgecolor='none', 
                            rcount=rcount, ccount=ccount, antialiased=True, shade=True)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    
    label_font = {
        'fontsize': cfg_3d.get("label_size", 14),
        'fontweight': cfg_3d.get("font_weight", "normal"),
        'family': cfg_3d.get("font_family", plt.rcParams["font.family"])
    }
    cbar.set_label("Max Shear Modulus (GPa)", labelpad=15, **label_font)

    max_val = np.max([np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))])
    limit = max_val * 1.3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.set_xlabel("X (GPa)", labelpad=15, **label_font)
    ax.set_ylabel("Y (GPa)", labelpad=15, **label_font)
    ax.set_zlabel("Z (GPa)", labelpad=15, **label_font)
    
    ax.tick_params(axis='both', which='major', labelsize=label_font['fontsize']-2)
    
    elev = cfg_3d.get("view_elevation", 20)
    azim = cfg_3d.get("view_azimuth", 45)
    ax.view_init(elev=elev, azim=azim)

    filename = build_filename(prefix, "Shear_Modulus_3D", output_format)
    gen = style_config.get("general", {}) if style_config else {}
    dpi = gen.get("dpi", 300)
    transparent = gen.get("transparent", True)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.8, transparent=transparent)
    if show:
        plt.show()
    plt.close()
    print(f"Saved 3D Shear Modulus plot to {filename}")
