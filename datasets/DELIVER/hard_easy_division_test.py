import os

# Directory containing the images /media/data4/sora/dataset/DELIVER_2/samples/images/test/
image_dir = 'PATH TO DELIVER TEST SET'

# Lists to hold filenames
easy_files = []
hard_files = []

# Iterate over files in the directory
for filename in os.listdir(image_dir):
    if 'underexposure' in filename:
        hard_files.append(filename)
    else:
        easy_files.append(filename)

# Write easy files to easy.txt
with open(os.path.join(image_dir.replace('/test',''), 'test_easy.txt'), 'w') as f:
    for file in easy_files:
        f.write(file + '\n')

# Write hard files to hard.txt
with open(os.path.join(image_dir.replace('/test',''), 'test_hard.txt'), 'w') as f:
    for file in hard_files:
        f.write(file + '\n')