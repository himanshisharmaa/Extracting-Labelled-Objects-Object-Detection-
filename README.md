## Extracting Labelled Objects for Object Detection

The Extracting of labelled objects involves an automated process to extract labeled objects from annotated images to enhance the efficiency of object detection tasks. This approach involves cropping each object of interest (e.g., drones) from the original image based on bounding box annotations, saving these cropped objects as individual images, and updating their corresponding labels accordingly.

Advantages:
- **Enhanced Model Training**: By isolating objects and focusing only on relevant features, we ensure that object detection models are trained on more precise data, leading to improved accuracy.
- **Efficient Data Handling**: Cropped images reduce file sizes and computational overhead, optimizing both storage and processing time.
- **Label Consistency**: The process automatically updates the annotations to match the cropped image regions, ensuring that each object is correctly labeled for further use in model training.
- **Increased Dataset Augmentation**: Extracted objects can be easily augmented (e.g., through rotation, scaling, etc.), allowing for a more diverse and robust dataset for training.
