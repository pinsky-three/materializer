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
from typing import List, Tuple, Optional, NamedTuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
RESIZE_DIM = (150, 150)
KMEANS_CLUSTERS = 5
KMEANS_MAX_ITER = 100
KMEANS_N_INIT = 3
KMEANS_BATCH_SIZE = 1024
KMEANS_MAX_NO_IMPROVEMENT = 10
CANVAS_PADDING = 100
FINAL_PADDING = 500
OUTPUT_DIR = "result"
DATA_DIR = "data"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


# --- Result Types ---
class ImageLoadResult(NamedTuple):
    image: Optional[Image.Image] = None
    warning: Optional[str] = None
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        return self.image is not None and self.error is None


class ColorResult(NamedTuple):
    colors: Optional[List[Tuple[int, int, int, int]]] = None
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        return self.colors is not None and self.error is None


def process_image(image_path: str) -> bool:
    """Processes a single image: load, find colors, create padded version, save."""
    logger.debug(f"Attempting to process: {image_path}")

    # 1. Load and Convert Image
    load_result = load_convert_image(image_path)

    match load_result:
        case ImageLoadResult(success=True, image=im, warning=warn_msg):
            if warn_msg:
                logger.warning(f"{warn_msg} - Image: {image_path}")
            logger.info(f"Successfully loaded image {image_path}: {im.size}")
            # Proceed to color extraction

        case ImageLoadResult(success=False, error=err):
            logger.error(f"Failed to load/convert image {image_path}: {err}", exc_info=isinstance(err, Exception))
            return False # Indicate failure

        case _: # Catch unexpected cases
             logger.error(f"Unexpected result from load_convert_image for {image_path}")
             return False

    # 2. Extract Dominant Colors (only if load succeeded)
    color_result = extract_dominant_colors(im) # type: ignore <im is guaranteed to be non-None here>

    match color_result:
        case ColorResult(success=True, colors=palette):
             if not palette: # Should be caught by error handling, but double-check
                  logger.error(f"Color extraction returned success but no palette for {image_path}")
                  return False
             logger.debug(f"Extracted palette for {image_path} (showing top 1): {palette[0]}")
             # Proceed to image manipulation

        case ColorResult(success=False, error=err):
            logger.error(f"Failed to extract dominant colors for {image_path}: {err}", exc_info=isinstance(err, Exception))
            return False # Indicate failure

        case _: # Catch unexpected cases
            logger.error(f"Unexpected result from extract_dominant_colors for {image_path}")
            return False

    # 3. Image Manipulation (only if load and color extraction succeeded)
    try:
        # Use the second most common color as background, or first if only one exists
        background_color = palette[1] if len(palette) > 1 else palette[0] # type: ignore <palette is guaranteed non-empty list>
        logger.debug(f"Using background color: {background_color} for {image_path}")

        x, y = im.size # type: ignore <im is guaranteed non-None>
        canvas_size = (x + 2 * CANVAS_PADDING, y + 2 * CANVAS_PADDING)
        canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0)) # Transparent canvas
        # Use alpha mask (im) only if the image has alpha channel ('RGBA')
        paste_mask = im if im.mode == 'RGBA' else None # type: ignore
        canvas.paste(im, (CANVAS_PADDING, CANVAS_PADDING), mask=paste_mask)

        w, h = canvas.size
        final_side = max(w, h) + FINAL_PADDING
        final_image = Image.new("RGBA", (final_side, final_side), background_color)

        offset_x, offset_y = (final_side - w) // 2, (final_side - h) // 2
        final_image.paste(canvas, (offset_x, offset_y), mask=canvas) # Use canvas alpha

    except Exception as e:
        logger.error(f"Error during image manipulation for {image_path}: {e}", exc_info=True)
        return False

    # 4. Save Result
    try:
        relative_path = os.path.relpath(image_path, DATA_DIR)
        result_dir = os.path.join(OUTPUT_DIR, os.path.dirname(relative_path))
        result_filename = os.path.splitext(os.path.basename(image_path))[0] + "_result.png"
        result_path = os.path.join(result_dir, result_filename)

        os.makedirs(result_dir, exist_ok=True)
        logger.info(f"Saving result to {result_path}")
        final_image.save(result_path, format="PNG") # Explicitly specify format
        return True # Indicate success

    except ValueError: # Handles case where image_path might not be under DATA_DIR
        logger.warning(f"Could not determine relative path for {image_path} from '{DATA_DIR}'. Attempting to save in '{OUTPUT_DIR}' root.")
        try:
            result_filename = os.path.splitext(os.path.basename(image_path))[0] + "_result.png"
            result_path = os.path.join(OUTPUT_DIR, result_filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            logger.info(f"Saving result to {result_path}")
            final_image.save(result_path, format="PNG")
            return True
        except Exception as e:
            logger.error(f"Failed to save result for {image_path} even in root: {e}", exc_info=True)
            return False
    except Exception as e:
        logger.error(f"Failed to save result {result_path}: {e}", exc_info=True)
        return False


def load_convert_image(image_path: str) -> ImageLoadResult:
    """Opens, loads, and converts an image to RGBA, handling potential errors."""
    try:
        with Image.open(image_path) as im_raw:
            exif_warning_msg: Optional[str] = None
            try:
                # Explicitly load data which triggers exif processing
                im_raw.load()
            except TypeError as e:
                tb_str = "".join(traceback.format_tb(e.__traceback__))
                # Check if it's the specific recoverable EXIF error
                if "expected string or bytes-like object, got 'tuple'" in str(e) and 're/__init__.py' in tb_str and 'exif_transpose' in tb_str:
                    exif_warning_msg = f"Problematic EXIF data: {e}. Proceeding without automatic transposition."
                else:
                    # Different, likely fatal, TypeError during load
                    logger.debug(f"Non-EXIF TypeError during load for {image_path}", exc_info=True)
                    return ImageLoadResult(error=e)
            except Exception as e:
                 # Other error during load
                 logger.debug(f"Exception during load for {image_path}", exc_info=True)
                 return ImageLoadResult(error=e)

            # Try converting, potentially after recoverable EXIF error
            im = im_raw.convert("RGBA")
            return ImageLoadResult(image=im, warning=exif_warning_msg)

    except FileNotFoundError:
        # Error during Image.open or .convert
        logger.debug(f"File not found: {image_path}")
        return ImageLoadResult(error=FileNotFoundError(f"File not found: {image_path}"))
    except Exception as e:
        # Error during Image.open or .convert
        logger.debug(f"Exception during open/convert for {image_path}", exc_info=True)
        return ImageLoadResult(error=e)


def extract_dominant_colors(image: Image.Image) -> ColorResult:
    """Extracts dominant colors using KMeans clustering."""
    try:
        if image.mode not in ['RGB', 'RGBA']:
            image_rgb = image.convert('RGB')
        else:
            image_rgb = image

        # Use LANCZOS for potentially better resize quality
        image_small = image_rgb.resize(RESIZE_DIM, resample=Image.Resampling.LANCZOS)
        ar = asarray(image_small)

        if ar.size == 0 or ar.shape[0] * ar.shape[1] == 0:
             return ColorResult(error=ValueError("Image array is empty or invalid after processing."))

        # Ensure we only use RGB for clustering
        if ar.shape[2] == 4:
             ar = ar[:, :, :3]
        elif ar.shape[2] != 3:
             return ColorResult(error=ValueError(f"Unexpected array shape {ar.shape}."))

        shape = ar.shape
        ar_reshaped = ar.reshape(prod(shape[:2]), shape[2]).astype(float)

        if ar_reshaped.shape[0] == 0:
            return ColorResult(error=ValueError("Image array is empty after reshape."))

        n_samples = ar_reshaped.shape[0]
        n_clusters = min(KMEANS_CLUSTERS, n_samples)

        if n_clusters <= 0: # Changed check to <= 0
            return ColorResult(error=ValueError("Not enough samples for clustering."))

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=KMEANS_MAX_ITER,
            random_state=1000,
            n_init=KMEANS_N_INIT,
            batch_size=min(KMEANS_BATCH_SIZE, n_samples), # Prevent batch size > samples
            max_no_improvement=KMEANS_MAX_NO_IMPROVEMENT,
            compute_labels=True
        ).fit(ar_reshaped)

        codes = kmeans.cluster_centers_
        vecs = kmeans.labels_

        counts, _bins = histogram(vecs, bins=range(n_clusters + 1))

        # Filter out potential NaN or Inf values in codes
        valid_indices = [i for i, code in enumerate(codes) if all(c == c and abs(c) != float('inf') for c in code)]
        if not valid_indices:
             return ColorResult(error=ValueError("No valid cluster centers found after filtering."))

        codes = codes[valid_indices]
        counts = counts[valid_indices]

        dominant_colors_list: List[Tuple[int, int, int, int]] = []
        sorted_indices = argsort(counts)[::-1]
        for index in sorted_indices:
            color_rgb = tuple(max(0, min(255, int(round(c)))) for c in codes[index])
            dominant_colors_list.append(color_rgb + (255,)) # Add alpha

        if not dominant_colors_list:
            return ColorResult(error=ValueError("No colors determined after processing clusters."))

        return ColorResult(colors=dominant_colors_list)

    except Exception as e:
        logger.debug(f"Exception during dominant color extraction", exc_info=True)
        return ColorResult(error=e)


def main():
    processed_count = 0
    failed_count = 0 # Renamed from error_count for clarity

    if not os.path.isdir(DATA_DIR):
        logger.error(f"Data directory '{DATA_DIR}' not found. Exiting.")
        return

    logger.info(f"Starting image processing in directory: {DATA_DIR}")

    for root, _, files in walk(DATA_DIR):
        for file in files:
            # Case-insensitive check for extensions
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                image_path = os.path.join(root, file)
                if process_image(image_path):
                    processed_count += 1
                else:
                    failed_count += 1 # Count images that failed processing

    logger.info(f"Processing complete. Successful: {processed_count}, Failed/Skipped: {failed_count}.")


if __name__ == "__main__":
    main()
