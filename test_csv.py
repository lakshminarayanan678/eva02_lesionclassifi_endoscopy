import os
import pandas as pd

# Define paths
root_dir = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/capsulevision/testing/Images/"
output_csv = "/home/endodl/PHASE-1/mln/lesions_cv24/MAIN/data1/capsulevision/capsulevision_metadata.csv"

records = []

# Loop through test images
for img_file in os.listdir(root_dir):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join('capsulevision', 'testing', 'Images', img_file)
        records.append({
            'dataset': 'capsulevision',
            'patient_id': '',
            'frame_path': img_path.replace('\\', '/'),
            'proposed_name': img_file,
            'class': '',
            'fold': -1,
            'original_class': ''
        })

# Save as CSV
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"Test metadata saved to {output_csv}")
