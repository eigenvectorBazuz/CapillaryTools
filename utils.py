import os
import cv2
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator

def display_comparison_with_annotator(root_dir, class_names=None, yolo_results=None, box_color=(0, 255, 0)):
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




def display_bboxes_with_global_id(root_dir, global_id, class_names=None):
    """
    Displays an image with its ground truth bounding boxes and serial numbers, based on global ID.

    Args:
    - root_dir (str): Root directory containing 'images', 'bboxes', and 'file_list.txt'.
    - global_id (int): Global ID (line number in file_list.txt) to load the corresponding image and bbox.
    - class_names (list): Optional list of class names corresponding to class IDs.
                          If None, it defaults to using class IDs.

    The bounding boxes are in a custom format: <class>, <confidence>, <top left x>, <top left y>, <width>, <height>
    """
    # Load the filenames from file_list.txt
    filenames = load_file_list(root_dir)

    # Check if the global_id is valid
    if global_id < 0 or global_id >= len(filenames):
        print(f"Invalid global ID: {global_id}. Must be between 0 and {len(filenames)-1}.")
        return

    # Get the corresponding filename (without extension) from file_list.txt
    file_name = filenames[global_id]

    # Paths to the image and corresponding bbox file
    img_path = os.path.join(root_dir, 'images', file_name + '.png')
    bbox_path = os.path.join(root_dir, 'bboxes', file_name + '.txt')

    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Image {file_name}.png not found in {root_dir}/images")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB for displaying color annotations

    # Check if class names are provided, otherwise use the class IDs
    class_names = class_names if class_names else ['0', '1']

    # Read the bbox file
    if not os.path.exists(bbox_path):
        print(f"Bounding box file {file_name}.txt not found in {root_dir}/bboxes")
        return

    with open(bbox_path, 'r') as f:
        bboxes = f.readlines()

    # Loop through each bbox and plot it on the image
    for i, bbox in enumerate(bboxes):
        parts = bbox.strip().split()
        if len(parts) != 6:
            print(f"Invalid format in bbox file for {file_name}.txt")
            continue

        class_id, confidence, x, y, width, height = parts
        class_id = int(class_id)
        x, y, width, height = map(float, [x, y, width, height])

        # Calculate bottom-right corner from top-left corner
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + width), int(y + height)

        # Draw the bounding
        if class_id == 0:
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw the class name and confidence near the box
        label = f"{class_names[class_id]} {confidence}"
        cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw the serial number near the box (a bit below the box)
        cv2.putText(img_rgb, f"#{i}", (x1, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
    
    
import os

def load_file_list(root_dir):
    """
    Loads the file_list.txt and returns a list of filenames (without extensions).
    """
    file_list_path = os.path.join(root_dir, 'file_list.txt')
    with open(file_list_path, 'r') as f:
        filenames = [line.strip() for line in f]
    return filenames

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
