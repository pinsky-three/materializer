# Materializer

This project processes images to extract their dominant colors and apply a padding and background fill around the image, saving the modified images in a specified output directory.

## Features

- Extracts dominant colors from an image using k-means clustering.
- Applies padding around the image.
- Sets a background color based on the second most dominant color.
- Saves processed images to a specified output directory.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/materializer.git
   cd materializer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure you have the required packages installed. This project relies on:
   - `Pillow`
   - `numpy`
   - `scipy`
   - `scikit-learn`

## Usage

1. Place images you wish to process in the `data` directory.
2. Run the main script:

   ```bash
   python main.py
   ```

   Processed images will be saved in the `result` directory.

### Example

- Original image located in `data` folder.
- Processed image with background padding and dominant color saved in `result`.

## Functions

- `process_image(image_path: str)`: Opens and processes an image, extracts dominant colors, adds padding, sets background, and saves the modified image.
- `dominant_colors(image: Image)`: Uses k-means clustering to determine the dominant colors in an image.
- `get_dominant_color(img: Image)`: Returns the most prominent color of an image.
- `hilo(a, b, c)`: Helper function for color calculations.
- `complement(r, g, b)`: Calculates the complementary color for the RGB values provided.

## Contributing

Feel free to fork this repository and submit pull requests with improvements. Please ensure that any contributions are well-documented and tested.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
