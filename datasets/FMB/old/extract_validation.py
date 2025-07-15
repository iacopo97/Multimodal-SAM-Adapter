import os
import random
import shutil

def copy_random_files(src_folder, dest_folder, num_files):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    files = os.listdir(src_folder)
    random_files = random.sample(files, num_files)
    if "easy" in src_folder:
        type = "easy"
    else:
        type = "hard"
    for file in random_files:
        with open(f'val_{type}_files.txt', 'a') as f:
            f.write(f"{file}\n")
        src_file = os.path.join(src_folder, file)
        dest_file = os.path.join(dest_folder, file)
        shutil.copy(src_file, dest_file)
        os.remove(src_file)

# Define source and destination folders
src_easy = r'C:\\Users\\iacop\\Desktop\\ar\\work\\FMB\\drive-download-20241030T120255Z-001\\FMB\\train\\Visible\\easy'
dest_easy = r'C:\\Users\\iacop\\Desktop\\ar\\work\\FMB\\drive-download-20241030T120255Z-001\\FMB\\val\\easy'

src_hard = r'C:\\Users\\iacop\\Desktop\\ar\\work\\FMB\\drive-download-20241030T120255Z-001\\FMB\\train\\Visible\\hard'
dest_hard = r'C:\\Users\\iacop\\Desktop\\ar\\work\\FMB\\drive-download-20241030T120255Z-001\\FMB\\val\\Visible\\hard'

# Copy 80 random files from each source folder to the corresponding destination folder
copy_random_files(src_easy, dest_easy, 80)
copy_random_files(src_hard, dest_hard, 80)