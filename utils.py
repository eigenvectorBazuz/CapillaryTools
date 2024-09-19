import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator

def display_comparison_with_annotator(root_dir, class_names=None, yolo_results=None, box_color=(0, 255, 0), overlay=False):
    """
    Displays two images side-by-side: 
    - Left: Ground truth bounding boxes
    - Right: YOLO prediction bounding boxes using Annotator from ultralytics.
    Allows custom box colors.
    """

    # file_name = filenames[global_id]
    file_name = yolo_results.path
    # img_path = os.path.join(root_dir, 'images', file_name + '.png')
    # bbox_path = os.path.join(root_dir, 'bboxes', file_name + '.txt')
    img_path = file_name
    bbox_path = file_name.replace('images', 'bboxes').replace('png', 'txt')

    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image {file_name}.png not found in {root_dir}/images")
        return

    img_rgb_gt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GT image copy
    if overlay:
        img_rgb_yolo = img_rgb_gt
    else:
        img_rgb_yolo = img_rgb_gt.copy()                    # YOLO image copy

    # 1. Display Ground Truth Boxes
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            bboxes = f.readlines()

        for i, bbox in enumerate(bboxes):
            class_id, confidence, x, y, width, height = map(float, bbox.strip().split())
            x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
            color = (255, 0, 0) if int(class_id) == 0 else (0, 0, 255)
            cv2.rectangle(img_rgb_gt, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_rgb_gt, f"GT #{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        print(f"No ground truth bounding box file found for {file_name}.txt")

    # 2. YOLO Predictions using Annotator (from previous saved results)
    for r in yolo_results:
        annotator = Annotator(img_rgb_yolo)  # Initialize Annotator with image

        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Get box coordinates (left, top, right, bottom)
            c = box.cls      # Class label
            label = f"{class_names[int(c)]} {box.conf[0]:.2f}"  # Class label with confidence
            # Use the custom box color
            annotator.box_label(b, label, color=box_color)

    # Display the result image with YOLO predictions
    img_with_annotations = annotator.result()  

    # Plot side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(cv2.cvtColor(img_rgb_gt, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Ground Truth")
    ax[0].axis('off')
    
    ax[1].imshow(cv2.cvtColor(img_with_annotations, cv2.COLOR_BGR2RGB))
    ax[1].set_title("YOLO Predictions with Annotator")
    ax[1].axis('off')

    plt.show()




def find_class_1_indices(root_dir):
    """
    Finds the global indices of images that contain at least one bounding box with class 1.

    Args:
    - root_dir (str): Root directory containing 'images', 'bboxes', and 'file_list.txt'.

    Returns:
    - A list of global indices (starting from 0) of images that contain at least one class 1 bounding box.
    """
    filenames = load_file_list(root_dir)
    class_1_indices = []
    class_1_filenames = []

    # Loop through each file in the list
    for global_id, file_name in enumerate(filenames):
        bbox_path = os.path.join(root_dir, 'bboxes', file_name + '.txt')

        # Read the bounding box file
        if not os.path.exists(bbox_path):
            print(f"Bounding box file {file_name}.txt not found in {root_dir}/bboxes")
            continue

        with open(bbox_path, 'r') as f:
            bboxes = f.readlines()

        # Check if any bbox in the file has class 1
        contains_class_1 = False
        for bbox in bboxes:
            parts = bbox.strip().split()
            if len(parts) != 6:
                print(f"Invalid format in bbox file for {file_name}.txt")
                continue

            class_id = int(parts[0])  # Get the class ID (first element in the line)

            if class_id == 1:
                contains_class_1 = True
                break

        # If the file contains a class 1 bounding box, add its global index
        if contains_class_1:
            class_1_indices.append(global_id)
            class_1_filenames.append(file_name)

    return class_1_indices, class_1_filenames

# helper function to access the db and retrieve the gt
def get_gt_bboxes(root_dir, yolo_results):
    file_name = yolo_results.path
    bbox_path = file_name.replace('images', 'bboxes').replace('png', 'txt')

    # Load the third to sixth columns (index 2 to 5 in zero-based indexing) into a numpy array
    bboxes = np.loadtxt(bbox_path, usecols=(2, 3, 4, 5))

    return bboxes

