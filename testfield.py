from dataset import default_box_generator, COCO
import matplotlib.pyplot as plt
import numpy as np
class_num = 4 #cat dog person background


def display_image_with_restored_boxes(image, ann_box, boxs_default):
    """
    Display an image with restored bounding boxes drawn on it.

    Parameters:
        image (numpy.ndarray): The image to display.
        ann_box (numpy.ndarray): Array of bounding boxes in relative [dx, dy, dw, dh] format.
        boxs_default (numpy.ndarray): Default boxes with [px, py, pw, ph].
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image.transpose(1, 2, 0).astype(np.uint8))  # [C, H, W] to [H, W, C]
    plt.axis("off")

    img_height, img_width = image.shape[1], image.shape[2]

    for idx, (box, confidence) in enumerate(zip(ann_box, ann_confidence)):
        if confidence[-1] == 0:
            dx, dy, dw, dh = box  # relative coordinates from ann_box
            px, py, pw, ph = boxs_default[idx][:4]  # default box center and size from boxs_default

            # Restore absolute bounding box coordinates
            gx = pw * dx + px
            gy = ph * dy + py
            gw = pw * np.exp(dw)
            gh = ph * np.exp(dh)

            # Convert to top-left corner and bottom-right corner for plotting
            x_min = gx * img_width
            y_min = gy * img_height
            x_max = (gx + gw / 2) * img_width
            y_max = (gy + gh / 2) * img_height
            # 2 153.75 67.12 157.29 252.47
            print(f"Box {idx}: gx={gx}, gy={gy}, gw={gw}, gh={gh}")
            print(f" - Absolute coords: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

            # Draw the bounding box
            plt.gca().add_patch(
                plt.Rectangle((x_min, y_min), width=x_max - x_min, height=y_max - y_min,
                              fill=False, edgecolor="red", linewidth=2)
            )

    plt.show()

if __name__ == '__main__':
    boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])

    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False,
                        image_size=320)
    image, ann_box, ann_confidence = dataset[2]
    display_image_with_restored_boxes(image, ann_box, ann_confidence)