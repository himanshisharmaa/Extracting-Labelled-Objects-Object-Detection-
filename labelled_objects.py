import os 
import argparse
import json
from PIL import Image
import numpy as np
from xml.etree import ElementTree as ET

def crop_and_save_object(image_path, bbox, save_img_dir, obj_class, image_id, obj_id):
    # Open image and convert to NumPy array
    image = Image.open(image_path)
    image_np = np.array(image)

    # Convert bbox (x_min, y_min, width, height) to slicing coordinates
    x_min, y_min, width, height = map(int, bbox)
    x_max = x_min + width
    y_max = y_min + height

    # Validate the bounding box
    if x_min < 0 or y_min < 0 or x_max > image_np.shape[1] or y_max > image_np.shape[0]:
        print(f"Invalid bbox for image {image_path}: {bbox}")
        return None, 0, 0  # Return None to indicate failure

    # Slice the image array to crop
    cropped_image_np = image_np[y_min:y_max, x_min:x_max]

    # Check if the cropped image is empty
    if cropped_image_np.size == 0:
        print(f"Cropped image is empty for {image_path} with bbox {bbox}")
        return None, 0, 0  # Return None to indicate failure

    # Convert cropped NumPy array back to PIL image
    cropped_image = Image.fromarray(cropped_image_np)

    # Save the cropped image
    obj_filename = os.path.join(save_img_dir, f"{image_id}_obj{obj_id}_{obj_class.strip()}.jpg")
    print(f"Saving cropped image: {obj_filename}")
    cropped_image.save(obj_filename)
    
    # Return the saved filename and new width/height
    return obj_filename, cropped_image.width, cropped_image.height

# COCO processing function
def process_coco(annotation_file, image_dir, output_dir):
    with open(annotation_file, 'r') as file:
        coco_data = json.load(file)

    save_img_dir = os.path.join(output_dir, "Cropped_images")
    save_lbl_dir = os.path.join(output_dir, "Cropped_labels")

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    new_annotations = []
    new_images = []

    for obj_id, ann in enumerate(coco_data['annotations']):
        image_id = ann['image_id']
        bbox = ann['bbox']
        category_id = ann['category_id']

        image_filename = image_id_to_filename[image_id]
        image_path = os.path.join(image_dir, image_filename)

        # Crop and save the object using slicing
        obj_filename, new_width, new_height = crop_and_save_object(image_path, bbox, save_img_dir, category_id, image_id, obj_id)

        if obj_filename is not None:  # Only process if cropping was successful
            # Create new image entry for COCO format
            new_image_id = len(new_images) + 1
            new_images.append({
                "license": 4,  # Assign appropriate license if available
                "file_name": os.path.basename(obj_filename),
                "height": new_height,
                "width": new_width,
                "id": new_image_id
            })

            # Save the updated annotation with adjusted bbox
            new_annotations.append({
                "image_id": new_image_id,
                "category_id": category_id,
                "bbox": [0, 0, new_width, new_height],  # Adjusted bbox for cropped image
                "file_name": obj_filename
            })

    # Save the new COCO annotation file
    new_coco_annotation_file = os.path.join(save_lbl_dir, "cropped_coco_annotations.json")
    coco_data['images'] = new_images
    coco_data['annotations'] = new_annotations
    with open(new_coco_annotation_file, 'w') as file:
        json.dump(coco_data, file, indent=4)

def process_yolo(annotation_dir, image_dir, output_dir):
    save_img_dir = os.path.join(output_dir, "Cropped_images")
    save_lbl_dir = os.path.join(output_dir, "Cropped_labels")

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    for image_file in images:
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file)
        annotation_file = os.path.join(annotation_dir, f"{image_id}.txt")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as file:
                lines = file.readlines()

            image = Image.open(image_path)
            image_width, image_height = image.size

            for obj_id, line in enumerate(lines):
                # Read class index and YOLO coordinates
                class_id, x_center, y_center, width, height = map(float, line.split())

                # Convert YOLO normalized values to pixel coordinates
                x_min = int((x_center - width / 2) * image_width)
                y_min = int((y_center - height / 2) * image_height)
                x_max = int((x_center + width / 2) * image_width)
                y_max = int((y_center + height / 2) * image_height)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

                # Crop and save the object using slicing
                obj_filename, new_width, new_height = crop_and_save_object(image_path, bbox, save_img_dir, class_id, image_id, obj_id)

                if obj_filename is not None:  # Only process if cropping was successful
                    # Calculate normalized YOLO values for the cropped image
                    new_x_center = 0.5  # Since the cropped image starts at (0, 0)
                    new_y_center = 0.5
                    new_width_norm = new_width / new_width  # This will always be 1.0
                    new_height_norm = new_height / new_height  # This will always be 1.0

                    # Generate new label filename based on cropped image
                    new_label_file_name = os.path.splitext(os.path.basename(obj_filename))[0] + ".txt"
                    new_label_file_path = os.path.join(save_lbl_dir, new_label_file_name)

                    # Append updated YOLO annotations
                    new_annotation = f"{int(class_id)} {new_x_center} {new_y_center} {new_width_norm} {new_height_norm}\n"

                    # Write new annotation file for each cropped image
                    with open(new_label_file_path, 'w') as label_file:
                        label_file.write(new_annotation)

# VOC processing function
def process_voc(annotation_dir, image_dir, output_dir):
    save_img_dir = os.path.join(output_dir, "Cropped_images")
    save_lbl_dir = os.path.join(output_dir, "Cropped_labels")

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in images:
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_dir, image_file.strip())
        annotation_file = os.path.join(annotation_dir, f"{image_id}.xml")

        if os.path.exists(annotation_file):
            tree = ET.parse(annotation_file)
            root = tree.getroot()

            # Get the image's width and height from the annotation file
            size = root.find('size')
            original_width = int(size.find('width').text)
            original_height = int(size.find('height').text)

            for obj_id, obj in enumerate(root.findall('object')):
                print()
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                obj_class = obj.find('name').text

                # Convert to width and height format for cropping (use tuple only for cropping)
                crop_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

                # Crop and save object
                cropped_image_path, new_width, new_height = crop_and_save_object(
                    image_path, crop_bbox, save_img_dir, obj_class, image_id, obj_id)

                if cropped_image_path is not None:  # Only proceed if cropping was successful
                    # Update the bounding box relative to the cropped image (starting at (0, 0))
                    bbox.find('xmin').text = str(0)
                    bbox.find('ymin').text = str(0)
                    bbox.find('xmax').text = str(new_width)
                    bbox.find('ymax').text = str(new_height)

                    # Update the size for the cropped image
                    size.find('width').text = str(new_width)
                    size.find('height').text = str(new_height)
                    root.find("filename").text=os.path.basename(cropped_image_path)
                    # Save updated annotation for the cropped image
                    new_voc_annotation_file = os.path.join(save_lbl_dir, f"{image_id}_obj{obj_id}_{obj_class.strip()}.xml")
                    tree.write(new_voc_annotation_file)


# General function to process dataset
def process_dataset(image_dir, annotation_path, annotation_type, output_dir):
    if annotation_type == "COCO":
        process_coco(annotation_path, image_dir, output_dir)
    elif annotation_type == "VOC":
        process_voc(annotation_path, image_dir, output_dir)
    elif annotation_type == "YOLO":
        process_yolo(annotation_path, image_dir, output_dir)
    else:
        raise ValueError(f"Unsupported annotation type: {annotation_type}")

# Main function for argument parsing and starting the process
if __name__ == "__main__":
    dir = os.getcwd()
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="Path to the image directory")
    ap.add_argument("-a", "--annotation_path", required=True, help="Path to the annotation directory or file")
    ap.add_argument("-o", "--output_dir", default=os.path.join(dir, "Cropped_Outputs/"), help="Path to the output directory")
    args = vars(ap.parse_args())
    
    image_dir = args["image_dir"]
    annotation_path = args["annotation_path"]
    output_dir = args["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(annotation_path):
        annotation_type = "COCO"
    elif os.path.isdir(annotation_path):
        files = os.listdir(annotation_path)
        if files[0].endswith(".txt"):
            annotation_type = "YOLO"
        elif files[0].endswith(".xml"):
            annotation_type = "VOC"
    else:
        raise ValueError(f"Unknown annotation format: {annotation_path}")

    process_dataset(image_dir, annotation_path, annotation_type, output_dir)
