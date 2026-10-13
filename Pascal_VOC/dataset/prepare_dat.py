# prepare_dataset_single_object.py

import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from scipy.ndimage import center_of_mass
import random # ### NEW IMPORT ###

# --- CONFIGURATION ---
KAGGLE_DATA_ROOT = '/bhupender/datasets/voc_2012_segmentation_data/train'
OUTPUT_DIR = './pascal_sam_dataset_final/'
NUM_IMAGES = 500  # Total images you want in your final dataset

CLASS_NAMES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
    'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
]
COLORS = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]
COLOR_MAP = {name: tuple(color) for name, color in zip(CLASS_NAMES, COLORS)}

# --- List of classes you are interested in ---
TARGET_CLASSES = ['person']

def prepare_dataset_and_prompts(source_dir, out_dir, num_images, target_classes):
    target_colors = {cls: COLOR_MAP[cls] for cls in target_classes if cls in COLOR_MAP}
    if not target_colors:
        print(f"Error: None of the target classes found in color map.")
        return

    final_image_dir = os.path.join(out_dir, target_classes[0], 'images')
    final_mask_dir = os.path.join(out_dir, target_classes[0], 'masks')
    os.makedirs(final_image_dir, exist_ok=True)
    os.makedirs(final_mask_dir, exist_ok=True)

    source_image_dir = os.path.join(source_dir, 'images')
    source_mask_dir = os.path.join(source_dir, 'masks')
    all_masks = os.listdir(source_mask_dir)

    prompts_data = {}
    selected_count = 0

    print(f"Searching for {num_images} images containing any of {target_classes}...")
    # Shuffle the masks to get a random assortment of images
    random.shuffle(all_masks)
    
    for mask_filename in tqdm(all_masks, desc="Processing data"):
        if selected_count >= num_images:
            break

        mask_path = os.path.join(source_mask_dir, mask_filename)
        color_mask_np = np.array(Image.open(mask_path).convert('RGB'))

        # ### MODIFIED LOGIC STARTS HERE ###

        # 1. Find all target classes that are present in this specific image
        present_classes = []
        for cls, color in target_colors.items():
            if np.any(np.all(color_mask_np == color, axis=-1)):
                present_classes.append(cls)

        # 2. If at least one of our target classes was found...
        if present_classes:
            
            # 3. ...randomly select ONLY ONE of them to be the subject for this sample.
            chosen_class = random.choice(present_classes)
            chosen_color = COLOR_MAP[chosen_class]

            # 4. Create the binary mask for ONLY the chosen class
            mask_logic = np.all(color_mask_np == chosen_color, axis=-1)
            binary_mask_np = (mask_logic * 255).astype(np.uint8)

            # 5. Generate the single point prompt for the chosen class
            # Use the center of mass of the object as the prompt
            if np.count_nonzero(binary_mask_np) == 0:
                continue  # skip if no object found
            com_y, com_x = center_of_mass(binary_mask_np)
            point_prompt = [int(round(com_x)), int(round(com_y))]

            # 6. Save the single, clean mask using the original filename
            mask_save_path = os.path.join(final_mask_dir, mask_filename)
            Image.fromarray(binary_mask_np).save(mask_save_path)

            # 7. Copy the corresponding original image
            image_filename = mask_filename.replace('.png', '.jpg')
            shutil.copy(os.path.join(source_image_dir, image_filename), os.path.join(final_image_dir, image_filename))
            
            # 8. Store the prompt data for this single object
            # prompts_data[image_filename] = {
            #     "points": [point_prompt], # List containing one point
            #     "labels": [1],            # List containing one label
            #     "class": chosen_class     # Note which class was chosen for this image
            # }
            
            selected_count += 1
        
        # ### MODIFIED LOGIC ENDS HERE ###

    if prompts_data:
        prompts_json_path = os.path.join(out_dir, 'prompts.json')
        with open(prompts_json_path, 'w') as f:
            json.dump(prompts_data, f, indent=4)
        print(f"\nSuccessfully generated 'prompts.json' at '{prompts_json_path}'")

    print(f"\nFinished! Created a dataset with {selected_count} single-object images in '{out_dir}'")

if __name__ == '__main__':
    # It's a clear, sunny morning here in Kattangal, perfect for some data processing.
    if not os.path.exists(KAGGLE_DATA_ROOT):
        print(f"Error: Could not find the Kaggle dataset at '{KAGGLE_DATA_ROOT}'")
    else:
        prepare_dataset_and_prompts(KAGGLE_DATA_ROOT, OUTPUT_DIR, NUM_IMAGES, TARGET_CLASSES)