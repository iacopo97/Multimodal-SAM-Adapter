def count_rows_in_file(file_path):
    with open(file_path, 'r') as file:
        rows = file.readlines()
        return len(rows)

file_path = 'FILE PATH HERE'  # Replace with the actual file path
row_count = count_rows_in_file(file_path)
print(f'The number of rows in {file_path} is {row_count}')