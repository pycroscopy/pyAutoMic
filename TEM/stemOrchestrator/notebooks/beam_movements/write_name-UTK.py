from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *
import autoscript_tem_toolkit.vision as vision_toolkit
import time
import numpy as np

def write_utk_with_beam_positioning():
    """
    Write 'UTK' using paused scan beam positioning in STEM mode and capture HAADF image
    """
    # Connect to microscope
    microscope = TemMicroscopeClient()
    microscope.connect("10.46.217.241", 9095)
    microscope.optics.blanker.unblank()
    
    try:
        # Switch to STEM mode
        print("Switching to STEM mode...")
        microscope.optics.optical_mode = OpticalMode.STEM
        
        # Set up beam parameters for writing
        print("Setting up beam parameters...")
        microscope.optics.scan_field_of_view = 16.4e-9  # 83 nanometer scan field
        microscope.optics.blanker.unblank()
        
        # Define letter dimensions (in relative coordinates 0.0 to 1.0)
        letter_height = 0.3  # 30% of field of view
        letter_width = 0.15   # 15% of field of view
        letter_spacing = 0.08   # 8% spacing between letters
        
        # Define writing parameters
        dwell_time = 1  # seconds per position (adjust for desired thickness)
        step_size = 0.005  # 0.5% of field of view per step
        
        print("Starting to write 'UTK' with beam positioning...")
        
        # Calculate starting position for centering (3 letters)
        total_width = letter_width * 3 + letter_spacing * 2
        start_x = 0.5 - total_width / 2  # Center horizontally
        start_y = 0.5 - letter_height / 2  # Center vertically
        
        # Write letter 'U'
        # print("Writing letter 'U'...")
        # write_letter_U_with_beam(microscope, start_x, start_y, 
        #                        letter_height, letter_width, dwell_time, step_size)
        
        # Write letter 'T'
        print("Writing letter 'T'...")
        t_start_x = start_x + letter_width + letter_spacing
        write_letter_T_with_beam(microscope, t_start_x, start_y,
                               letter_height, letter_width, dwell_time, step_size)
        
        # # Write letter 'K'
        # print("Writing letter 'K'...")
        # k_start_x = t_start_x + letter_width + letter_spacing
        # write_letter_K_with_beam(microscope, k_start_x, start_y,
        #                        letter_height, letter_width, dwell_time, step_size)
        
        print("Finished writing 'UTK'")
        
        # Wait a moment before imaging
        time.sleep(1)
        
        # Acquire HAADF image to see the result
        # print("Acquiring HAADF image of written 'UTK'...")
        # image = microscope.acquisition.acquire_stem_image(
        #     DetectorType.HAADF, 
        #     ImageSize.PRESET_2048, 
        #     dwell_time=1e-6  # Fast imaging dwell time
        # )
        
        # # Save and display the image
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # filename = f'UTK_beam_writing_{timestamp}.tiff'
        # image.save(filename)
        # vision_toolkit.plot_image(image)
        
        # print(f"Image saved as '{filename}'")
        print("Script completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Disconnect from microscope
        microscope.disconnect()

def write_letter_U_with_beam(microscope, start_x, start_y, height, width, dwell, step_size):
    """Write letter 'U' using paused scan beam positioning"""
    
    # Define U shape coordinates (relative to letter bounds)
    u_path = [
        (0.0, 0.0),    # Top left
        (0.0, 0.75),   # Left side down
        (0.15, 0.95),  # Bottom left curve
        (0.5, 1.0),    # Bottom center
        (0.85, 0.95),  # Bottom right curve
        (1.0, 0.75),   # Right side up
        (1.0, 0.0),    # Top right
    ]
    
    # Convert to absolute field coordinates and draw
    draw_path_with_beam(microscope, u_path, start_x, start_y, 
                       height, width, dwell, step_size)

def write_letter_T_with_beam(microscope, start_x, start_y, height, width, dwell, step_size):
    """Write letter 'T' using paused scan beam positioning"""
    
    # Define T shape coordinates - horizontal top bar + vertical line
    t_path = [
        (0.0, 0.0),    # Top left
        (1.0, 0.0),    # Top right (horizontal bar)
    ]
    
    # Draw horizontal bar
    draw_path_with_beam(microscope, t_path, start_x, start_y,
                       height, width, dwell, step_size)
    
    # Draw vertical line (centered)
    vertical_path = [
        (0.5, 0.0),    # Top center
        (0.5, 1.0),    # Bottom center
    ]
    
    draw_path_with_beam(microscope, vertical_path, start_x, start_y,
                       height, width, dwell, step_size)

def write_letter_K_with_beam(microscope, start_x, start_y, height, width, dwell, step_size):
    """Write letter 'K' using paused scan beam positioning"""
    
    # Define K shape coordinates - vertical line + two diagonal lines
    # Vertical line
    vertical_path = [
        (0.0, 0.0),    # Top left
        (0.0, 1.0),    # Bottom left
    ]
    
    draw_path_with_beam(microscope, vertical_path, start_x, start_y,
                       height, width, dwell, step_size)
    
    # Upper diagonal (from middle left to top right)
    upper_diagonal = [
        (0.0, 0.5),    # Middle left
        (1.0, 0.0),    # Top right
    ]
    
    draw_path_with_beam(microscope, upper_diagonal, start_x, start_y,
                       height, width, dwell, step_size)
    
    # Lower diagonal (from middle left to bottom right)
    lower_diagonal = [
        (0.0, 0.5),    # Middle left
        (1.0, 1.0),    # Bottom right
    ]
    
    draw_path_with_beam(microscope, lower_diagonal, start_x, start_y,
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

def write_utk_advanced_with_beam():
    """
    Advanced version with better control and thicker lines
    """
    microscope = TemMicroscopeClient()
    microscope.connect("10.46.217.241", 9095)
    
    try:
        # STEM setup
        microscope.optics.optical_mode = OpticalMode.STEM
        microscope.optics.scan_field_of_view = 16.4e-9  # 83 nanometer field
        microscope.optics.blanker.unblank()
        
        # Enhanced writing parameters for thicker lines
        base_dwell = 0.1  # Base dwell time
        line_passes = 3   # Multiple passes for thicker lines
        pass_offset = 0.003  # Slight offset between passes
        
        print("Writing 'UTK' with enhanced thickness...")
        
        # Calculate positions for 3 letters
        letter_width = 0.15
        letter_spacing = 0.08
        total_width = letter_width * 3 + letter_spacing * 2
        base_start_x = 0.5 - total_width / 2
        base_start_y = 0.35
        
        # Write with multiple passes for thickness
        for pass_num in range(line_passes):
            offset = (pass_num - line_passes//2) * pass_offset
            
            print(f"Pass {pass_num + 1}/{line_passes}")
            
            # Write U with offset
            write_letter_U_with_beam(microscope, base_start_x + offset, base_start_y + offset, 
                                   0.3, letter_width, base_dwell, 0.003)
            
            # Write T with offset
            t_x = base_start_x + letter_width + letter_spacing
            write_letter_T_with_beam(microscope, t_x + offset, base_start_y + offset,
                                   0.3, letter_width, base_dwell, 0.003)
            
            # Write K with offset
            k_x = t_x + letter_width + letter_spacing
            write_letter_K_with_beam(microscope, k_x + offset, base_start_y + offset,
                                   0.3, letter_width, base_dwell, 0.003)
        
        # Final image acquisition
        # print("Acquiring final HAADF image...")
        # image = microscope.acquisition.acquire_stem_image(
        #     DetectorType.HAADF, ImageSize.PRESET_2048, 1e-6)
        
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        # filename = f'UTK_thick_beam_writing_{timestamp}.tiff'
        # image.save(filename)
        # vision_toolkit.plot_image(image)
        
        # print(f"Image saved as '{filename}'")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        
    finally:
        microscope.disconnect()

if __name__ == "__main__":
    print("Choose writing method:")
    print("1. Standard beam positioning")
    print("2. Advanced thick line writing")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        write_utk_advanced_with_beam()
    else:
        write_utk_with_beam_positioning()