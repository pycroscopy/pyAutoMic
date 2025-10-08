# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>


from typing import Tuple, List
import numpy as np

def serialize_array(array: np.ndarray) -> Tuple[List, Tuple, str]:
    """
    Serializes a numpy array into a list format with metadata.
    
    Args:
        array (np.ndarray): Input numpy array to serialize
    
    Returns:
        Tuple[List, Tuple, str]: Tuple containing:
            - List: Array data as nested Python lists
            - Tuple: Shape of the original array
            - str: Data type of the original array
    """
    array_list = array.tolist()
    dtype = str(array.dtype)
    return array_list, array.shape, dtype


class DMtwin:
    """Emulates the Gatan Digital Micrograph PC for EELS acquisition."""
    
    def __init__(self) -> None:
        """Initialize the DM simulator."""
        pass
    
    def acquire_camera(self, exposure = 0.05) -> None: # mimicks the eels detector
        pass
    
    def get_eels(self) -> Tuple[List[float], Tuple[int], str]:
        """
        Generate simulated EELS spectrum data with a prominent plasmon peak.
        
        Returns:
            Tuple[List[float], Tuple[int], str]: Serialized EELS data containing:
                - List[float]: 1D spectrum intensities (1028 channels)
                - Tuple[int]: Shape (1028,)
                - str: Data type
        """
        # Create energy loss axis (0 to 100 eV)
        x = np.linspace(0, 100, 1028)
        
        # Zero loss peak (sharper)
        zlp = 5000 * np.exp(-(x - 0)**2 / (2 * 0.3**2))
        
        # Prominent plasmon peak (snail hump)
        plasmon = 2000 * np.exp(-(x - 15)**2 / (2 * 8**2))
        
        # Background decay (tail)
        background = 500 * np.exp(-x/30)
        
        # Combine components
        eels = zlp + plasmon + background
        
        # Add subtle noise
        noise = np.random.normal(0, np.sqrt(eels + 1) * 0.1)
        eels = eels + noise
        
        # Ensure non-negative values
        eels = np.maximum(eels, 0)
        eels = (eels - np.min(eels)) / (np.max(eels) - np.min(eels))
        
        return serialize_array(eels)
    
    
