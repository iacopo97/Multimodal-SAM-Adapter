import os

def list_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def write_list_to_file(file_list, file_path):
    with open(file_path, 'w') as file:
        for item in file_list:
            file.write(f"{item}\n")

def main():
    easy_folder = 'C:\\Users\\iacop\\Desktop\\ar\\work\\FMB\\drive-download-20241030T120255Z-001\\test\\test\\Visible\\easy'
    hard_folder = 'C:\\Users\\iacop\\Desktop\\ar\\work\\FMB\\drive-download-20241030T120255Z-001\\test\\test\\Visible\\hard'
    
    easy_files = list_files_in_folder(easy_folder)
    hard_files = list_files_in_folder(hard_folder)
    
    write_list_to_file(easy_files, 'easy_files.txt')
    write_list_to_file(hard_files, 'hard_files.txt')

if __name__ == "__main__":
    main()