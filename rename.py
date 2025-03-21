import os


def rename_files(directory, prefix):
    # List all files in the directory
    files = os.listdir(directory)

    for index, filename in enumerate(files):
        # Construct the new filename
        if index < 9:
            new_filename = f"{prefix}_0{index + 1}{os.path.splitext(filename)[1]}"
        else:
            new_filename = f"{prefix}_{index + 1}{os.path.splitext(filename)[1]}"
        # Get the full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_filename}'")


# Example usage
directory = r"dataset\adhd_test"
prefix = "adhd_test"
rename_files(directory, prefix)
