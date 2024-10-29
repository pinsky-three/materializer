from PIL import Image  # , ImageOps
from os import walk
from numpy import asarray, prod, histogram, argsort
from scipy import cluster
from sklearn.cluster import MiniBatchKMeans
# from colorsys import rgb_to_hsv, hsv_to_rgb


def process_image(image_path: str):
    with Image.open(image_path).convert("RGBA") as im:
        print(f"Image {image_path} : {im.size}")

        palette = dominant_colors(im)

        most_common_color = palette[1]
        # print(f"most_common_color: {most_common_color}")
        # most_common_color = rgb_to_hsv(*most_common_color[:-1])
        # print(f"most_common_color hsv: {most_common_color}")

        # most_common_color = (
        #     (most_common_color[0] + 0.5) % 1,
        #     most_common_color[1] + 0.1,
        #     most_common_color[2] + 0.01,
        # )

        # most_common_color = hsv_to_rgb(*most_common_color)
        # print(f"most_common_color rgb: {most_common_color}")
        # most_common_color = (
        #     int(most_common_color[0]),
        #     int(most_common_color[1]),
        #     int(most_common_color[2]),
        #     255,
        # )

        x, y = im.size
        pad = 100

        size = (x + 2 * pad, y + 2 * pad)

        canvas = Image.new("RGBA", size, (255, 255, 255, 255))
        canvas.paste(im, (pad, pad), im)

        w, h = canvas.size
        side = max(w, h) + 500

        final = Image.new("RGBA", (side, side), most_common_color)

        x, y = (side - w) // 2, (side - h) // 2
        final.paste(canvas, (x, y), canvas)

        result_path = image_path.replace("data", "result").split(".")[0] + "_result.png"
        final.save(result_path)


def get_dominant_color(img: Image):
    img = img.copy()
    img = img.convert("RGBA")
    img = img.resize((1, 1), resample=0)
    dominant_color = img.getpixel((0, 0))
    return dominant_color


def hilo(a, b, c):
    if c < b:
        b, c = c, b
    if b < a:
        a, b = b, a
    if c < b:
        b, c = c, b
    return a + c


def complement(r, g, b):
    k = hilo(r, g, b)
    return (*tuple(k - u for u in (r, g, b)), 255)


def dominant_colors(image: Image):
    image = image.resize((150, 150))

    ar = asarray(image)
    shape = ar.shape
    ar = ar.reshape(prod(shape[:2]), shape[2]).astype(float)

    kmeans = MiniBatchKMeans(
        n_clusters=5, init="k-means++", max_iter=20, random_state=1000
    ).fit(ar)

    codes = kmeans.cluster_centers_

    vecs, _dist = cluster.vq.vq(ar, codes)  # assign codes
    counts, _bins = histogram(vecs, len(codes))  # count occurrences

    colors = []
    for index in argsort(counts)[::-1]:
        colors.append(tuple([int(code) for code in codes[index]]))
    return colors  # returns colors in order of dominance


def main():
    for root, _, files in walk("data"):
        for file in files:
            if file.endswith((".jpg", ".tif", ".png")):
                process_image(f"{root}/{file}")
                # break


if __name__ == "__main__":
    main()
