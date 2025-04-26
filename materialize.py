from PIL import Image
from PIL import ImageEnhance 
from typing import Tuple


def materialize_image(
    original_image: Image.Image,
    background_color: Tuple[int, int, int, int],
    canvas_padding: int,
    final_padding: int
) -> Image.Image:
    curr_bri = ImageEnhance.Brightness(original_image) 
    new_bri = 1.2
    
    img_brightened = curr_bri.enhance(new_bri) 

    curr_col = ImageEnhance.Color(img_brightened) 
    new_col = 1.6

    img_colored = curr_col.enhance(new_col)

    curr_sharp = ImageEnhance.Sharpness(img_colored) 
    new_sharp = 2.15
    
    img_sharped = curr_sharp.enhance(new_sharp) 
    
     
    
    return img_sharped
