# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>


"""
Has:
    - haadf tiff plot with scalebar
    - drift correction : return shift in x and y
    - drift plotting

"""
import xml.etree.ElementTree as ET
from collections import defaultdict
import tifffile
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
import logging
import numpy as np
import skimage.registration
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Tuple, List
import numpy as np
import skimage.registration
from scipy import ndimage

def tiff_to_numpy(tiff_path: str) -> np.ndarray:
    """
    Reads a TIFF file and returns its image data as a NumPy array.

    Args:
        tiff_path (str): Path to the TIFF file.

    Returns:
        np.ndarray: The image data as a NumPy array.
    """
    with tifffile.TiffFile(tiff_path) as tif:
        return tif.asarray()

def HAADF_tiff_to_png(HAADF_path: str) -> None:
    """
    Autoscript HAADF tiff support
    Reads a HAADF TIFF file, extracts pixel size metadata, and plots the image with a scalebar.

    Args:
        HAADF_path (str): Path to the HAADF TIFF file.

    Returns:
        None
    """
    
    def etree_to_dict(element):
        """Converts an XML ElementTree into a nested dictionary."""
        d = {element.tag: {} if element.attrib else None}
        children = list(element)
        if children:
            dd = defaultdict(list)
            for dc in map(etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {element.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if element.attrib:
            d[element.tag].update(('@' + k, v) for k, v in element.attrib.items())
        if element.text:
            text = element.text.strip()
            if children or element.attrib:
                if text:
                    d[element.tag]['#text'] = text
            else:
                d[element.tag] = text
        return d

    # Open the TIFF file and extract metadata
    with tifffile.TiffFile(HAADF_path) as tif:
        # Check if FEI_TITAN metadata exists
        fei_titan_metadata_xml = tif.pages[0].tags.get('FEI_TITAN')
        if fei_titan_metadata_xml is None:
            print("Warning: FEI_TITAN metadata not found in the TIFF file.")
            return
        
        # Parse metadata
        root = ET.fromstring(fei_titan_metadata_xml.value)
        metadata_dict = etree_to_dict(root)

    # Extract the PixelSize information
    binary_result = metadata_dict.get('Metadata', {}).get('BinaryResult', {})
    pixel_size_x = binary_result.get('PixelSize', {}).get('X', {}).get('#text')
    unit_x = binary_result.get('PixelSize', {}).get('X', {}).get('@unit')
    
    # Convert pixel size to float
    try:
        pixel_size_x = float(pixel_size_x)
    except (TypeError, ValueError):
        print("Error: Could not extract valid pixel size from metadata.")
        return

    # Read the TIFF image data
    with tifffile.TiffFile(HAADF_path) as tif:
        image = tif.asarray()

    # Plot the image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    # Add a scale bar
    scalebar = ScaleBar(pixel_size_x, location='lower right', length_fraction=0.25,
                        label=f'1 pixel = {pixel_size_x:.2e} {unit_x}', color='white', box_alpha=0.5)
    ax.add_artist(scalebar)

    # Hide axes
    ax.axis('off')

    # Show the plot
    plt.show()

def drift_compute_GD(fixed_image: np.ndarray, shifted_image: np.ndarray) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Simple FFT based drift correction 
    # credits : https://github.com/pycroscopy/pyTEMlib/blob/main/pyTEMlib/image_tools.py -> def rigid_registration(dataset, sub_pixel=True):
    """
    fft_fixed = np.fft.fft2(fixed_image)
    fft_moving = np.fft.fft2(shifted_image)
    image_product = fft_fixed * fft_moving.conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    shift = np.array(ndimage.maximum_position(cc_image.real))-fixed_image.shape[0]/2

    return shift[0], shift[1]

def compute_drift(
    image1: np.ndarray,
    image2: np.ndarray,
    sub_pixel: bool = True,
    return_magnitude: bool = False,
    return_angle: bool = False
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Compute drift between two images using phase cross-correlation.

    Parameters
    ----------
    image1 : np.ndarray
        First image (512x512).
    image2 : np.ndarray
        Second image (512x512).
    sub_pixel : bool, optional
        If True, uses sub-pixel registration for more precise measurements.
    return_magnitude : bool, optional
        If True, returns the magnitude of the drift vector.
    return_angle : bool, optional
        If True, returns the angle of the drift vector in degrees.

    Returns
    -------
    Tuple[float, float] or Tuple[float, float, Optional[float], Optional[float]]
        (x, y) drift in pixels.
        If `return_magnitude` is True, appends drift magnitude.
        If `return_angle` is True, appends drift angle in degrees.
    """
    shift = skimage.registration.phase_cross_correlation(
        image1, image2, upsample_factor=100 if sub_pixel else 1
    )[0]

    x_drift, y_drift = shift[0], shift[1]

    if return_magnitude or return_angle:
        magnitude = np.sqrt(x_drift**2 + y_drift**2) if return_magnitude else None
        angle = np.degrees(np.arctan2(y_drift, x_drift)) if return_angle else None
        return x_drift, y_drift, magnitude, angle

    return x_drift, y_drift 

def plot_drift_comparison(image1, image2, shift_x, shift_y):
    """
    Visualize drift between two images with overlays and vector arrows.

    Parameters
    ----------
    image1, image2 : ndarray
        Original images (512x512).
    shift_x, shift_y : float
        Drift components.

    Returns
    -------
    fig : Matplotlib figure
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Overlay shifted image on original
    combined = np.zeros((image1.shape[0], image1.shape[1], 3))
    combined[..., 0] = image1 / np.max(image1)  # Red Channel
    combined[..., 1] = np.roll(image2 / np.max(image2), (int(shift_y), int(shift_x)), axis=(0, 1))  # Green Channel

    # Center of image
    center_y, center_x = image1.shape[0] // 2, image1.shape[1] // 2
    end_x, end_y = center_x + shift_x, center_y + shift_y

    # First plot: Reference Image
    axs[0].imshow(image1, cmap="gray")
    axs[0].set_title("Reference Image")
    axs[0].plot(center_x, center_y, "ro", markersize=10, label="Reference")
    axs[0].arrow(center_x, center_y, shift_x, shift_y, color="yellow", width=1, head_width=5)

    # Second plot: Shifted Image
    axs[1].imshow(image2, cmap="gray")
    axs[1].set_title("Shifted Image")
    axs[1].plot(end_x, end_y, "bo", markersize=10, label="Shifted")
    axs[1].arrow(end_x, end_y, -shift_x, -shift_y, color="yellow", width=1, head_width=5)

    # Third plot: Overlay (Red = Reference, Green = Shifted)
    axs[2].imshow(combined)
    axs[2].set_title("Overlay of Both Images (Drift Visualization)")

    # Add drift magnitude annotation
    plt.figtext(
        0.5,
        0.02,
        f"Measured Drift: ({shift_x:.2f}, {shift_y:.2f}) pixels | Magnitude: {np.sqrt(shift_x**2 + shift_y**2):.2f} pixels",
        ha="center",
        bbox=dict(facecolor="white", alpha=0.7),
    )

    plt.tight_layout()
    return fig



def plot_eels(eels: np.ndarray, 
              energy_range: tuple = (0, 1000),
              title: Optional[str] = None,
              save_path: Optional[str] = None) -> None:
    """
    Plot EELS spectrum with proper formatting.
    
    Args:
        eels (np.ndarray): EELS spectrum data
        energy_range (tuple): Energy loss range in eV
        title (str, optional): Plot title
        save_path (str, optional): Path to save figure
    """
    # Create energy loss axis
    energy_loss = np.linspace(energy_range[0], energy_range[1], len(eels))
    
    # Create figure with appropriate size
    plt.figure(figsize=(10, 6))
    
    # Plot spectrum
    plt.plot(energy_loss, eels, 'b-', linewidth=1)
    
    # Add labels and title
    plt.xlabel('Energy Loss (eV)')
    plt.ylabel('Intensity (a.u.)')
    if title:
        plt.title(title)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to log scale for better visibility of features
    # plt.yscale('log')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()