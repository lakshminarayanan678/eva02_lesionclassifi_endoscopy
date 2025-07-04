import os
import pandas as pd

# Path to the new testing dataset directory
new_dataset_root = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_eva02/testing"

# Output CSV file
output_csv_path = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/split_data_eva02/test.csv"

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Create list to collect rows
rows = []

# Classes
classes = ['Erosion', 'Ulcers', 'Normal']

for cls in classes:
    cls_dir = os.path.join(new_dataset_root, cls)
    if not os.path.isdir(cls_dir):
        continue
    for img_file in os.listdir(cls_dir):
        if img_file.endswith(image_extensions):
            full_path = os.path.join(cls_dir, img_file)
            row = {
                "dataset": "capsulevision",
                "patient_id": "",
                "frame_path": full_path,
                "proposed_name": img_file,
                "class": cls,
                "fold": -1,
                "original_class": ""
            }
            rows.append(row)

# Convert to DataFrame
df = pd.DataFrame(rows, columns=["dataset", "patient_id", "frame_path", "proposed_name", "class", "fold", "original_class"])

# Save to CSV
df.to_csv(output_csv_path, index=False)
print(f"Updated CSV saved to: {output_csv_path}")
