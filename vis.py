from matplotlib import pyplot as plt 
import matplotlib.cm as cm
import numpy as np

def plot_boxes(boxes):
  """Plots 2D bounding boxes with a color sequence for every 4 boxes.

  Args:
    boxes: A NumPy array (n x 4) where each row represents a box (x_min, y_min, x_max, y_max).
  """

  fig, ax = plt.subplots(figsize=(40,30))
  cmap = cm.get_cmap('hsv', 5)  # Colormap with 4 colors

  # Loop over boxes with a step of 4 (considering 4 boxes per color)
  for i in range(0, len(boxes), 4):
    # Get the current colormap
    colors = cmap(np.linspace(0, 1, 5))

    for j, box in enumerate(boxes[i:i+4]):
      x_min, y_min, x_max, y_max = box[4:]
      rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=2, edgecolor=colors[j], facecolor='none')
      ax.add_patch(rect)

  plt.axis('scaled')
  plt.savefig("output.png")
  plt.show()