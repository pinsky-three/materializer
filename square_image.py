from PIL import Image
from typing import Tuple


def generate_final_image(
    original_image: Image.Image,
    background_color: Tuple[int, int, int, int],
    canvas_padding: int,
    final_padding: int
) -> Image.Image:
    """Creates a new image with the original centered on a padded background."""
    im = original_image

    # 1. Create transparent canvas larger than the image
    x, y = im.size
    canvas_size = (x + 2 * canvas_padding, y + 2 * canvas_padding)
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    # 2. Paste original image onto the canvas center
    paste_mask = im if im.mode == 'RGBA' else None
    canvas.paste(im, (canvas_padding, canvas_padding), mask=paste_mask)

    # 3. Create final background image
    w, h = canvas.size
    final_side = max(w, h) + final_padding
    final_image = Image.new("RGBA", (final_side, final_side), background_color)

    # 4. Paste the canvas (with image) onto the final background center
    offset_x, offset_y = (final_side - w) // 2, (final_side - h) // 2
    final_image.paste(canvas, (offset_x, offset_y), mask=canvas)

    return final_image
