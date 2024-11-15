from PIL import ImageDraw
from PIL import Image

def draw_annotation_box():
    image_path = "data/test/images/00708.jpg"
    image = Image.open(image_path)

    annotation_path = "results/00708.txt"
    with open(annotation_path, "r") as file:
        line = file.readline().strip()
        class_id, xmin, ymin, w, h = map(float, line.split())
    xmax = xmin + w
    ymax = ymin + h
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    image.show()

if __name__ == '__main__':
    draw_annotation_box()
