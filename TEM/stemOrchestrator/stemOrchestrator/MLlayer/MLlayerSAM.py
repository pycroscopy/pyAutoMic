# Author credits - Utkarsh Pratiush <utkarshp1161@gmail.com>


import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os
import requests
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from datetime import datetime
from matplotlib.patches import Polygon


def setup_device() -> torch.device:
    """Set up and return the available device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def download_sam_model(
    model_type: str, checkpoint_url: str, checkpoint_path: str
) -> None:
    """Download the SAM model checkpoint if it doesn't exist."""
    if not os.path.exists(checkpoint_path):
        print("Downloading SAM model checkpoint...")
        response = requests.get(checkpoint_url)
        with open(checkpoint_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("SAM model checkpoint already exists.")


def initialize_sam_model(
    model_type: str, checkpoint_path: str, device: torch.device
) -> Tuple[Any, Any]:
    """Initialize and return the SAM model and mask generator."""
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam, mask_generator


def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB format and ensure correct dtype."""
    if not isinstance(image_data, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    if image_data.ndim != 2:  # Ensure it's a grayscale image (2D array)
        raise ValueError("Input image must be a 2D grayscale array")

    rgb_image = (
        np.stack((image_data,) * 3, axis=-1).astype(np.float32) / 255.0
    )  # Normalize to [0,1]
    return rgb_image


def generate_and_save_masks(
    mask_generator: Any, img_np: np.ndarray, output_path: str
) -> List[Dict[str, Any]]:
    """Generate masks using SAM, validate input, handle errors, and save to disk."""

    # Validate inputs
    if not isinstance(img_np, np.ndarray):
        raise TypeError("img_np must be a NumPy array")

    if img_np.ndim not in (2, 3):  # Check for grayscale (2D) or RGB (3D)
        raise ValueError("img_np must be a 2D grayscale or 3D RGB image array")

    try:
        print("Generating masks...")
        masks = mask_generator.generate(img_np)

        if not masks or not isinstance(masks, list):
            raise ValueError("Mask generation failed or returned an unexpected format")

        print(f"Number of masks generated: {len(masks)}")

        # Save masks to disk
        with open(output_path, "wb") as f:
            pickle.dump(masks, f)

        return masks

    except Exception as e:
        print(f"Error during mask generation: {e}")
        return []


def create_visualization_with_masks(
    img_np: np.ndarray, masks: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
    """Create an image with colored masks and return centroids."""
    visual_image = img_np.copy()
    centroids = []

    # Iterate through each mask and overlay it with a unique color
    for idx, mask in enumerate(masks, 1):
        segmentation = mask["segmentation"]
        # Generate a random color
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        # Create a colored mask
        colored_mask = np.zeros_like(visual_image)
        colored_mask[segmentation] = color
        # Blend the colored mask with the original image
        visual_image = cv2.addWeighted(visual_image, 1.0, colored_mask, 0.5, 0)

        # Compute centroid
        coords = np.column_stack(np.where(segmentation))
        if coords.size == 0:
            continue  # Skip if mask is empty
        centroid = coords.mean(axis=0)
        centroids.append((centroid[1], centroid[0], idx))  # (x, y, label)

    return visual_image, centroids


def display_image_with_masks(visual_image: np.ndarray, title: str) -> None:
    """Display image with colored masks."""
    plt.figure(figsize=(8, 8))
    plt.imshow(visual_image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def display_image_with_labels(
    visual_image: np.ndarray, centroids: List[Tuple[float, float, int]], title: str
) -> None:
    """Display image with colored masks and labels at centroids."""
    plt.figure(figsize=(8, 8))
    plt.imshow(visual_image)
    ax = plt.gca()

    for x, y, label in centroids:
        # Choose a contrasting color for the text
        text_color = "white" if np.mean(visual_image[int(y), int(x)]) < 128 else "black"
        ax.text(
            x,
            y,
            str(label),
            color=text_color,
            fontsize=12,
            bbox=dict(
                facecolor="red" if text_color == "white" else "yellow", alpha=0.5, pad=1
            ),
        )

    plt.axis("off")
    plt.title(title)
    plt.show()


def extract_mask_contours(masks: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Extract contours for each mask."""
    mask_contours = {}

    for idx, mask in enumerate(masks, 1):
        segmentation = mask["segmentation"].astype(np.uint8)
        segmentation = segmentation * 255  # Convert boolean to 0-255
        contours, _ = cv2.findContours(
            segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # For simplicity, consider only the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify the contour to reduce the number of vertices
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            # Reshape for easier handling
            approx_contour = approx_contour.reshape(-1, 2)
            mask_contours[idx] = approx_contour
        else:
            mask_contours[idx] = None  # No contour found

    print(f"Extracted contours for {len(mask_contours)} masks.")
    return mask_contours


def generate_mask_colors(num_masks: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate random colors for each mask."""
    mask_colors = {}
    for idx in range(1, num_masks + 1):
        mask_colors[idx] = tuple([np.random.randint(0, 255) for _ in range(3)])
    return mask_colors


def visualize_masks_with_boundaries(
    image: np.ndarray,
    centroids: List[Tuple[float, float, int]],
    mask_contours: Dict[int, np.ndarray],
    mask_colors: Dict[int, Tuple[int, int, int]],
    output_path: str,
) -> None:
    """Visualize segmentation masks with their boundaries and centroids."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for x, y, label in centroids:
        contour = mask_contours.get(label)
        if contour is not None:
            # Create a polygon patch for the contour
            polygon = Polygon(
                contour,
                closed=True,
                linewidth=2,
                edgecolor=np.array(mask_colors[label]) / 255,
                fill=False,
            )
            ax.add_patch(polygon)

        # Overlay the centroid with label number
        pixel_brightness = np.mean(image[int(y), int(x)])
        text_color = "white" if pixel_brightness < 128 else "black"
        ax.text(
            x,
            y,
            str(label),
            color=text_color,
            fontsize=12,
            bbox=dict(
                facecolor="red" if text_color == "white" else "yellow", alpha=0.5, pad=1
            ),
        )

    plt.axis("off")
    plt.title("Segmentation Masks with Boundaries and Centroids")
    plt.savefig(output_path)
    plt.close()


def extract_particle_data(masks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract particle data including centroids and boundaries."""
    particles = []

    for idx, mask in enumerate(masks, 1):
        segmentation = mask["segmentation"]

        # Compute centroid
        coords = np.column_stack(np.where(segmentation))
        if coords.size == 0:
            continue  # Skip if mask is empty
        centroid = coords.mean(axis=0)  # (y, x)

        # Extract contours using OpenCV
        segmentation_uint8 = (segmentation * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            segmentation_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Consider only the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify the contour
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            # Reshape for easier handling
            boundary = approx_contour.reshape(-1, 2)
            num_boundary_points = boundary.shape[0]
        else:
            boundary = None
            num_boundary_points = 0

        # Append to particles list
        particles.append(
            {
                "label": idx,
                "centroid": (centroid[1], centroid[0]),  # (x, y)
                "boundary": boundary,
                "num_boundary_points": num_boundary_points,
            }
        )

    print(f"Total particles stored: {len(particles)}")
    return particles


def print_boundary_points_info(particles: List[Dict[str, Any]]) -> None:
    """Print information about boundary points per particle."""
    total_boundary_points = 0

    print("Boundary Points per Particle:")
    print("-----------------------------")

    for particle in particles:
        label = particle["label"]
        num_points = particle["num_boundary_points"]
        total_boundary_points += num_points
        print(f"Particle {label}: {num_points} boundary points")

    print("-----------------------------")
    print(f"Total boundary points across all particles: {total_boundary_points}")


def plot_centroids(centroids: np.ndarray, img_np: np.ndarray) -> None:
    """Plot the centroids of segmentation masks on the image."""
    plt.figure(figsize=(8, 8))
    plt.scatter(centroids[:, 0], centroids[:, 1], c=centroids[:, 2], s=2, cmap="tab20")
    plt.title(f"Centroids of Segmentation Masks -- total particles: {len(centroids)}")
    plt.imshow(img_np)
    plt.colorbar()
    plt.show()


def sample_particle_positions(
    particles: List[Dict[str, Any]],
    img_np: np.ndarray,
    sampling_percentage: float = 1.0,
) -> np.ndarray:
    """Sample particle positions including centroids and boundaries."""
    positions_list = []
    np.random.seed(42)  # Set a seed for reproducibility

    for particle in particles:
        label = particle["label"]
        centroid_x, centroid_y = particle["centroid"]

        # Encode the label for centroid
        centroid_label = (label * 10) + 0
        positions_list.append([centroid_x, centroid_y, centroid_label])

        boundary = particle["boundary"]

        if boundary is not None and len(boundary) > 0:
            total_boundary_points = len(boundary)
            num_sampled_points = max(
                1, int(sampling_percentage * total_boundary_points)
            )

            # Randomly sample boundary points without replacement
            sampled_indices = np.random.choice(
                range(total_boundary_points), num_sampled_points
            )
            sampled_boundary = boundary[sampled_indices]

            for point in sampled_boundary:
                x, y = point
                boundary_label = (label * 10) + 1
                positions_list.append([x, y, boundary_label])

    positions_sampled = np.array(positions_list, dtype=float)
    print(f"Positions array shape: {positions_sampled.shape}")

    return positions_sampled


def plot_sampled_positions(
    positions_sampled: np.ndarray, img_np: np.ndarray, num_centroids: int
) -> None:
    """Plot the sampled positions (centroids and boundary points) on the image."""
    plt.figure(figsize=(8, 8))
    plt.scatter(
        positions_sampled[:, 0],
        positions_sampled[:, 1],
        c=positions_sampled[:, 2],
        s=2,
        cmap="tab20",
    )
    plt.title(
        f"Centroids and sampled boundary points -- total particles: {num_centroids}"
    )
    plt.imshow(img_np)
    plt.colorbar()
    plt.show()


def create_normalized_particle_positions(
    particles: List[Dict[str, Any]],
    img_shape: Tuple[int, int],
    sampling_percentage: float = 1.0,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Create a dictionary of normalized particle positions by particle ID."""
    each_particle_position = {}
    np.random.seed(42)  # Set a seed for reproducibility

    for particle in particles:
        label = particle["label"]
        centroid_x, centroid_y = particle["centroid"]

        # Add the centroid as a NumPy array (normalized by image dimensions)
        each_particle_position[label] = {
            "centroid": np.array(
                [centroid_x / img_shape[0], centroid_y / img_shape[1]]
            ),
            "boundary_points": np.empty(
                (0, 2)
            ),  # Initialize empty array for boundary points
        }

        boundary = particle["boundary"]

        if boundary is not None and len(boundary) > 0:
            total_boundary_points = len(boundary)
            num_sampled_points = max(
                1, int(sampling_percentage * total_boundary_points)
            )

            # Randomly sample boundary points without replacement
            sampled_indices = np.random.choice(
                range(total_boundary_points), num_sampled_points, replace=False
            )
            # Normalize boundary points by image dimensions
            sampled_boundary = boundary[sampled_indices] / img_shape[0]

            # Store normalized boundary points
            each_particle_position[label]["boundary_points"] = np.array(
                sampled_boundary
            )

    return each_particle_position


def main(image_data: np.ndarray, path_folder: str) -> None:
    """Main function to run SAM segmentation pipeline."""
    # Setup
    device = setup_device()

    # Model configuration
    model_type = "vit_b"  # Options: 'vit_b', 'vit_l', 'vit_h'
    checkpoint_url = (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )
    checkpoint_path = "sam_vit_b_01ec64.pth"

    # Download and initialize model
    download_sam_model(model_type, checkpoint_url, checkpoint_path)
    sam, mask_generator = initialize_sam_model(model_type, checkpoint_path, device)

    # Preprocess image
    img_np = preprocess_image(image_data)

    # Display original image
    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    # Generate masks
    masks_path = f"{path_folder}/masks_Au_online.pkl"
    masks = generate_and_save_masks(mask_generator, img_np, masks_path)

    # Visualize masks
    visual_image, centroids = create_visualization_with_masks(img_np, masks)
    display_image_with_masks(visual_image, "Image with Segmentation Masks")
    display_image_with_labels(
        visual_image, centroids, "Image with Segmentation Masks and Labels"
    )

    # Extract and process contours
    mask_contours = extract_mask_contours(masks)
    mask_colors = generate_mask_colors(len(masks))

    # Visualize masks with boundaries
    boundaries_path = (
        f"{path_folder}/Segmentation Masks with Boundaries and Centroids.png"
    )
    visualize_masks_with_boundaries(
        visual_image, centroids, mask_contours, mask_colors, boundaries_path
    )

    # Extract particle data
    particles = extract_particle_data(masks)

    # Save particle data
    with open(f"{path_folder}/particles.pkl", "wb") as f:
        pickle.dump(particles, f)

    # Print boundary point information
    print_boundary_points_info(particles)

    # Plot centroids
    centroids_array = np.array(centroids)
    plot_centroids(centroids_array, img_np)

    # Sample particle positions
    positions_sampled = sample_particle_positions(particles, img_np)

    # Plot sampled positions
    plot_sampled_positions(positions_sampled, img_np, len(centroids))

    # Create normalized particle positions
    each_particle_position = create_normalized_particle_positions(
        particles, img_np.shape[:2]
    )

    # Save normalized particle positions
    with open(f"{path_folder}/sampled_boundary_pts_particles.pkl", "wb") as f:
        pickle.dump(each_particle_position, f)

    print("Processing complete!")


if __name__ == "__main__":
    # Example usage (uncomment and fill in your image data and path)
    import numpy as np
    from PIL import Image

    #
    # Load your image data
    image_path = "your_image_path.jpg"
    image_data = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale
    #
    # # Output folder
    path_folder = "output_folder"
    os.makedirs(path_folder, exist_ok=True)
    #
    main(image_data, path_folder)
