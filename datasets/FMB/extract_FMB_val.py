import os
import shutil

base_dir = os.path.dirname(os.path.abspath(__file__))

for data_division in ['train', 'val','test']:
    # Create easy and hard folders inside FMB/train/VISIBLE
    visible_dir = os.path.join(base_dir, data_division, 'Visible')
    for folder in ['easy', 'hard']:
        os.makedirs(os.path.join(visible_dir, folder), exist_ok=True)

    # Move files listed in train_easy_files.txt and train_hard_files.txt into respective folders
    for txt_file, target_folder in [(f'{data_division}_easy_files.txt', 'easy'), (f'{data_division}_hard_files.txt', 'hard')]:
        txt_path = os.path.join(base_dir, txt_file)
        with open(txt_path, 'r') as f:
            for line in f:
                filename = line.strip()
                if data_division == 'val':
                    temp_dir = visible_dir.replace('val', 'train')
                else:
                    temp_dir = visible_dir
                src_file = os.path.join(temp_dir, filename)
                dst_file = os.path.join(visible_dir, target_folder, filename)
                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)
                    # print(f"src: {src_file}, dst: {dst_file}")
                else:
                    print(f"Warning: {src_file} does not exist in VISIBLE.")
                
