# Key Advantages of Beam Positioning:

# Much faster: No mechanical stage movement delays
# Higher precision: Beam positioning is more accurate than stage movement
# Better control: Can easily control dwell time at each pixel
# No vibrations: Eliminates stage movement artifacts

# Key Features:

# Relative coordinates: Uses 0.0-1.0 coordinate system for the scan field
# Configurable parameters: Easy to adjust letter size, thickness, and spacing
# Multiple passes: Advanced version draws multiple passes for thicker lines
# Safety bounds: Ensures beam coordinates stay within [0.0, 1.0] range
# Step control: Adjustable step size for line smoothness

# Writing Parameters:

# dwell_time: 0.05-0.1 seconds per position (controls line darkness/thickness)
# step_size: 0.003-0.005 relative units (controls line smoothness)
# scan_field_of_view: 8-10 μm (total writing area)
# letter_height/width: 0.3 x 0.15 relative units

# Advanced Version Features:

# Multiple passes with slight offsets for thicker lines
# Enhanced dwell times
# Better line quality

# The beam will "paint" the letters by dwelling at each position, creating visible damage/contrast that will show up clearly in the HAADF image!










from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *
import autoscript_tem_toolkit.vision as vision_toolkit
import time
import numpy as np

def write_wd_with_beam_positioning():
    """
    Write 'WD' using paused scan beam positioning in STEM mode and capture HAADF image
    """
    # Connect to microscope
    microscope = TemMicroscopeClient()
    microscope.connect("10.1.149.210", 9090)
    microscope.optics.blanker.unblank()
    
    try:
        # Switch to STEM mode
        print("Switching to STEM mode...")
        microscope.optics.optical_mode = OpticalMode.STEM
        
        # Set up beam parameters for writing
        print("Setting up beam parameters...")
        # microscope.optics.spot_size_index = 1  # Small spot for precise writing
        # microscope.optics.convergence_angle = 0.025  # High convergence for focused beam
        microscope.optics.scan_field_of_view = 83e-9  # 10 micrometer scan field
        microscope.optics.blanker.unblank()
        
        # Define letter dimensions (in relative coordinates 0.0 to 1.0)
        letter_height = 0.3  # 30% of field of view
        letter_width = 0.2   # 20% of field of view
        line_thickness = 0.02  # Controlled by dwell time and beam current
        letter_spacing = 0.1   # 10% spacing between letters
        
        # Define writing parameters
        dwell_time = 0.05  # seconds per position (adjust for desired thickness)
        step_size = 0.005  # 0.5% of field of view per step
        
        print("Starting to write 'WD' with beam positioning...")
        
        # Calculate starting position for centering
        total_width = letter_width * 2 + letter_spacing
        start_x = 0.5 - total_width / 2  # Center horizontally
        start_y = 0.5 - letter_height / 2  # Center vertically
        
        # Write letter 'W'
        print("Writing letter 'W'...")
        write_letter_W_with_beam(microscope, start_x, start_y, 
                               letter_height, letter_width, dwell_time, step_size)
        
        # Write letter 'D'
        print("Writing letter 'D'...")
        d_start_x = start_x + letter_width + letter_spacing
        write_letter_D_with_beam(microscope, d_start_x, start_y,
                               letter_height, letter_width, dwell_time, step_size)
        
        print("Finished writing 'WD'")
        
        # Wait a moment before imaging
        time.sleep(1)
        
        # Acquire HAADF image to see the result
        print("Acquiring HAADF image of written 'WD'...")
        image = microscope.acquisition.acquire_stem_image(
            DetectorType.HAADF, 
            ImageSize.PRESET_2048, 
            dwell_time=1e-6  # Fast imaging dwell time
        )
        
        # Save and display the image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'WD_beam_writing_{timestamp}.tiff'
        image.save(filename)
        vision_toolkit.plot_image(image)
        
        print(f"Image saved as '{filename}'")
        print("Script completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Disconnect from microscope
        microscope.disconnect()

def write_letter_W_with_beam(microscope, start_x, start_y, height, width, dwell, step_size):
    """Write letter 'W' using paused scan beam positioning"""
    
    # Define W shape coordinates (relative to letter bounds)
    w_path = [
        (0.0, 0.0),    # Top left
        (0.25, 1.0),   # Bottom left leg
        (0.5, 0.4),    # Middle peak
        (0.75, 1.0),   # Bottom right leg
        (1.0, 0.0),    # Top right
    ]
    
    # Convert to absolute field coordinates and draw
    draw_path_with_beam(microscope, w_path, start_x, start_y, 
                       height, width, dwell, step_size)

def write_letter_D_with_beam(microscope, start_x, start_y, height, width, dwell, step_size):
    """Write letter 'D' using paused scan beam positioning"""
    
    # Define D shape coordinates - vertical line + curved right side
    d_path = [
        (0.0, 0.0),    # Top left
        (0.0, 1.0),    # Bottom left (vertical line)
        (0.4, 1.0),    # Bottom curve start
        (0.7, 0.9),    # Bottom curve
        (0.85, 0.7),   # Right side bottom
        (0.9, 0.5),    # Right side middle
        (0.85, 0.3),   # Right side top
        (0.7, 0.1),    # Top curve
        (0.4, 0.0),    # Top curve start
        (0.0, 0.0),    # Back to start
    ]
    
    # Convert to absolute field coordinates and draw
    draw_path_with_beam(microscope, d_path, start_x, start_y,
                       height, width, dwell, step_size)

def draw_path_with_beam(microscope, path_points, start_x, start_y, 
                       height, width, dwell_time, step_size):
    """Draw a path by positioning the paused scan beam"""
    
    # Convert path points to field coordinates
    field_points = []
    for rel_x, rel_y in path_points:
        field_x = start_x + rel_x * width
        field_y = start_y + rel_y * height
        field_points.append((field_x, field_y))
    
    # Draw each line segment
    for i in range(len(field_points) - 1):
        draw_line_with_beam(microscope, field_points[i], field_points[i + 1],
                           dwell_time, step_size)

def draw_line_with_beam(microscope, start_point, end_point, dwell_time, step_size):
    """Draw a line by moving the paused scan beam position"""
    
    start_x, start_y = start_point
    end_x, end_y = end_point
    
    # Calculate line parameters
    dx = end_x - start_x
    dy = end_y - start_y
    distance = np.sqrt(dx**2 + dy**2)
    
    # Calculate number of steps
    num_steps = max(int(distance / step_size), 1)
    
    # Calculate step increments
    dx_step = dx / num_steps if num_steps > 0 else 0
    dy_step = dy / num_steps if num_steps > 0 else 0
    
    print(f"Drawing line from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f}) in {num_steps} steps")
    
    # Move beam along the line with dwell at each position
    for step in range(num_steps + 1):
        beam_x = start_x + step * dx_step
        beam_y = start_y + step * dy_step
        
        # Ensure coordinates are within valid range [0.0, 1.0]
        beam_x = max(0.0, min(1.0, beam_x))
        beam_y = max(0.0, min(1.0, beam_y))
        
        # Position the paused scan beam
        microscope.optics.paused_scan_beam_position = [beam_x, beam_y]
        
        # Dwell at this position (this is where the "writing" happens)
        time.sleep(dwell_time)

def write_wd_advanced_with_beam():
    """
    Advanced version with better control and thicker lines
    """
    microscope = TemMicroscopeClient()
    microscope.connect()
    
    try:
        # STEM setup
        # microscope.optics.optical_mode = OpticalMode.STEM
        # microscope.optics.spot_size_index = 1
        microscope.optics.scan_field_of_view = 8e-6  # 8 micrometer field
        
        # Enhanced writing parameters for thicker lines
        base_dwell = 0.1  # Base dwell time
        line_passes = 3   # Multiple passes for thicker lines
        pass_offset = 0.003  # Slight offset between passes
        
        print("Writing 'WD' with enhanced thickness...")
        
        # Write with multiple passes for thickness
        for pass_num in range(line_passes):
            offset = (pass_num - line_passes//2) * pass_offset
            
            print(f"Pass {pass_num + 1}/{line_passes}")
            
            # Write W with offset
            write_letter_W_with_beam(microscope, 0.25 + offset, 0.35 + offset, 
                                   0.3, 0.15, base_dwell, 0.003)
            
            # Write D with offset  
            write_letter_D_with_beam(microscope, 0.55 + offset, 0.35 + offset,
                                   0.3, 0.15, base_dwell, 0.003)
        
        # Final image acquisition
        print("Acquiring final HAADF image...")
        image = microscope.acquisition.acquire_stem_image(
            DetectorType.HAADF, ImageSize.PRESET_2048, 1e-6)
        
        image.save('WD_thick_beam_writing.tiff')
        vision_toolkit.plot_image(image)
        
    finally:
        microscope.disconnect()

if __name__ == "__main__":
    print("Choose writing method:")
    print("1. Standard beam positioning")
    print("2. Advanced thick line writing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        write_wd_advanced_with_beam()
    else:
        write_wd_with_beam_positioning()