import glob
import os
from PIL import Image

SRC_PATH = os.path.join("..", "dataset", "afhq")
DST_PATH = os.path.join("..", "dataset", "cats_and_dogs_256x256")

for dataset in glob.glob(os.path.join(SRC_PATH, "*")):
    dataset_base = os.path.basename(dataset)
    for class_name in ["cat", "dog"]:
        dataset_class = os.path.join(dataset, class_name)
        target_dir = os.path.join(DST_PATH, dataset_base, class_name)
        os.makedirs(target_dir, exist_ok=True)
        for img in glob.glob(os.path.join(dataset_class, "*.jpg")):
            image_name = os.path.basename(img)
            image_path = os.path.join(target_dir, image_name)

            with Image.open(img) as image:
                image = image.resize((256, 256))
                image.save(image_path)
                print(f"Rescaled {img} to {image_path}")
            