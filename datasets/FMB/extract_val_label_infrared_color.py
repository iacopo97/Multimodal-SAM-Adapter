import os
import random
import shutil
# Define the source and destination directories
base_dir = os.path.dirname(os.path.abspath(__file__))

for data_type in ['Label','color','Infrared']:
    for easy_hard in ['easy', 'hard']:
        source_dir = base_dir+f'/train/{data_type}'
        validation_dir = base_dir+f'/val/Visible/{easy_hard}'
        destination_dir = base_dir+f'/val/{data_type}'

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Get the list of files in the validation directory
        validation_files = os.listdir(validation_dir)

        # Copy the files from the source directory to the destination directory
        for file_name in validation_files:
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)
            if os.path.exists(source_file):
                # print(f"Copying {source_file} to {destination_file}")
                shutil.copy(source_file, destination_file)
            os.remove(source_file)