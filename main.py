from PIL import Image, ImageOps
from os import walk
from numpy import asarray, prod, histogram, argsort
from scipy import cluster
from sklearn.cluster import MiniBatchKMeans
# from colorsys import rgb_to_hsv, hsv_to_rgb
import os
import logging
import re # Import re for the original error context, though not used in fix
import traceback # Import traceback for detailed error checking

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
logger = logging.getLogger(__name__)


def process_image(image_path: str):
    try:
        # Open the image first
        with Image.open(image_path) as im_raw:
            exif_error_occurred = False
            try:
                # Explicitly load data which triggers exif processing where the error might occur
                im_raw.load()
            except TypeError as e:
                # Check if it's the specific error from re.sub in exif_transpose
                tb_str = "".join(traceback.format_tb(e.__traceback__))
                if "expected string or bytes-like object, got 'tuple'" in str(e) and 're/__init__.py' in tb_str and 'exif_transpose' in tb_str:
                    logger.warning(f"Problematic EXIF data encountered in {image_path}: {e}. Attempting to proceed without transposition.")
                    exif_error_occurred = True # Mark that the specific error happened
                    # Do NOT return yet, try converting anyway in the next step
                else:
                    # It's a different TypeError during load, treat as fatal for this image
                    logger.error(f"An unexpected TypeError occurred while loading {image_path}: {e}")
                    return # Skip this image

            # Now, outside the inner try-except for load(), try converting
            # This might still fail if the internal state is bad after the EXIF error
            im = im_raw.convert("RGBA")
            if exif_error_occurred:
                 logger.info(f"Successfully converted {image_path} despite earlier EXIF TypeError.")

    except Exception as e:
        # Handle errors during open, the convert call above, or other unexpected issues
        # Log with traceback for better debugging
        logger.error(f"Failed to process image {image_path}: {e}", exc_info=True)
        return # Skip this image

    # --- If loading and converting succeeded (potentially with EXIF warning) ---
    logger.info(f"Processing image {image_path} : {im.size}")

    palette = dominant_colors(im)

    if not palette:
        logger.warning(f"Could not determine dominant colors for {image_path}. Skipping.")
        return

    # Use the second most common color as background (index 1)
    # Make sure palette has at least 2 colors, otherwise use the first or a default
    if len(palette) > 1:
        background_color = palette[1]
    elif palette: # Only one color found
         background_color = palette[0]
    else: # Should not happen due to check above, but as fallback
        background_color = (128, 128, 128, 255) # Default gray
    logger.debug(f"Using background color: {background_color}")

    x, y = im.size
    pad = 100

    size = (x + 2 * pad, y + 2 * pad)

    # Create canvas with a transparent background initially
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    # Paste the image onto the transparent canvas
    # Use alpha mask (im) only if the image has alpha channel ('RGBA')
    canvas.paste(im, (pad, pad), im if im.mode == 'RGBA' else None)

    w, h = canvas.size
    # Make padding around the canvas larger
    side = max(w, h) + 500

    # Create final image with the chosen background color
    final = Image.new("RGBA", (side, side), background_color)

    # Calculate position to center the canvas on the final image
    offset_x, offset_y = (side - w) // 2, (side - h) // 2
    # Paste the canvas (with the image) onto the final background
    # Use the canvas's alpha channel as the mask to handle transparency correctly
    final.paste(canvas, (offset_x, offset_y), canvas)


    # Calculate relative path from 'data' dir to preserve structure in 'result'
    try:
        relative_path = os.path.relpath(image_path, "data")
    except ValueError:
        # Handle cases where image_path might not be under "data" if script usage changes
        logger.warning(f"Could not determine relative path for {image_path} from 'data'. Using base name.")
        relative_path = os.path.basename(image_path)

    result_dir = os.path.join("result", os.path.dirname(relative_path))
    result_filename = os.path.splitext(os.path.basename(image_path))[0] + "_result.png"
    result_path = os.path.join(result_dir, result_filename)

    os.makedirs(result_dir, exist_ok=True)
    logger.info(f"Saving result to {result_path}")
    final.save(result_path)


# dominant_colors function enhanced for robustness
def dominant_colors(image: Image):
    try:
        # Ensure image has data and is RGB(A)
        image.load()
        if image.mode not in ['RGB', 'RGBA']:
            image_rgb = image.convert('RGB')
        else:
            image_rgb = image

        # Resize for performance
        image_small = image_rgb.resize((150, 150), resample=Image.Resampling.NEAREST) # Use explicit resampling

        ar = asarray(image_small) # Should be 3 channels now

        # Handle cases like fully transparent images becoming 1x1 or similar edge cases
        if ar.size == 0 or ar.shape[0] * ar.shape[1] == 0:
             logger.warning("Image array is empty or invalid after processing. Returning empty palette.")
             return []

        # Ensure we only use RGB for clustering
        if ar.shape[2] == 4: # If RGBA came through somehow
             ar = ar[:, :, :3]
        elif ar.shape[2] != 3:
             logger.warning(f"Unexpected array shape {ar.shape}. Returning empty palette.")
             return []


        shape = ar.shape
        # Reshape correctly
        ar_reshaped = ar.reshape(prod(shape[:2]), shape[2]).astype(float)

        # Check if reshaped array is empty after potential filtering/conversion issues
        if ar_reshaped.shape[0] == 0:
            logger.warning("Image array is empty after reshape. Returning empty palette.")
            return []

        # Use more robust KMeans settings
        n_clusters_desired = 5
        n_samples = ar_reshaped.shape[0]
        n_clusters = min(n_clusters_desired, n_samples) # Cannot have more clusters than samples

        if n_clusters == 0:
            logger.warning("Not enough samples for clustering. Returning empty palette.")
            return []

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=100, # Increase iterations
            random_state=1000,
            n_init=3, # Recommended to be >= 3
            batch_size=1024, # Adjust based on expected image sizes/memory
            max_no_improvement=10 # Stop early if no improvement
        ).fit(ar_reshaped)

        codes = kmeans.cluster_centers_
        vecs, _dist = cluster.vq.vq(ar_reshaped, codes)
        counts, _bins = histogram(vecs, len(codes))

        # Filter out potential NaN or Inf values in codes before converting to int
        valid_indices = [i for i, code in enumerate(codes) if all(c == c and abs(c) != float('inf') for c in code)]

        if not valid_indices:
             logger.warning("No valid cluster centers found after filtering. Returning empty palette.")
             return []

        # Keep only valid codes and their corresponding counts
        codes = codes[valid_indices]
        counts = counts[valid_indices]

        colors = []
        # Sort remaining valid counts
        sorted_indices = argsort(counts)[::-1]
        for index in sorted_indices:
            # Ensure color values are within valid range [0, 255] and are integers
            color_rgb = tuple(max(0, min(255, int(round(c)))) for c in codes[index]) # Use round before int
            # Append alpha channel (fully opaque)
            colors.append(color_rgb + (255,))

        if not colors:
            logger.warning("No colors determined after processing clusters. Returning empty palette.")
        return colors

    except Exception as e:
        # Include traceback for errors in this function
        logger.error(f"Error in dominant_colors: {e}", exc_info=True)
        return [] # Return empty list on error


# Unused functions remain below
# def get_dominant_color(img: Image): ...
# def hilo(a, b, c): ...
# def complement(r, g, b): ...


def main():
    processed_count = 0
    skipped_count = 0
    error_count = 0
    data_dir = "data"
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory '{data_dir}' not found. Exiting.")
        return

    logger.info(f"Starting image processing in directory: {data_dir}")

    for root, _, files in walk(data_dir):
        for file in files:
            # Case-insensitive check for extensions
            if file.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png")):
                image_path = os.path.join(root, file)
                logger.debug(f"Found image file: {image_path}")
                # Modify process_image to return status: True (success), False (skipped), None (error)
                # For now, just call it and handle exceptions here
                try:
                    # process_image now handles internal skips/errors and returns None
                    process_image(image_path)
                    # We can't easily count skips vs successes without modifying return value
                    # Assume processed if no exception bubbles up
                    processed_count += 1 # This count is approximate
                except Exception as e:
                    logger.error(f"Unhandled exception processing {image_path} in main loop: {e}", exc_info=True)
                    error_count += 1

    logger.info(f"Processing complete. Approximate counts: Processed={processed_count}, Errors={error_count}.")
    # Note: Skipped files due to EXIF errors are logged but not counted separately here.


if __name__ == "__main__":
    main()
