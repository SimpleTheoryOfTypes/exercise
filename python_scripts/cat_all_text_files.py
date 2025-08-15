import os
import argparse

# This script appends all the text files under a directory into a single text file.
# python cat_all_text_files.py <path-to-your-dir> __combined_output.txt

def append_text_files_to_output(directory_path, output_file, extensions=None):
    """
    Append the contents of all text files in a directory to an output file.
    
    Args:
        directory_path (str): Path to the directory to search.
        output_file (str): Path to the output file where contents will be appended.
        extensions (list): List of file extensions to include (e.g., ['.txt', '.cpp']). 
                          If None, defaults to common text file extensions.
    """
    # Default extensions if none provided
    if extensions is None:
        extensions = ['.txt', '.cpp', '.cu', '.h', '.py', '.md']

    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory '{directory_path}' does not exist.")

    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as outfile:
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(directory_path):
            for filename in files:
                # Check if the file has one of the specified extensions
                if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    file_path = os.path.join(root, filename)
                    try:
                        # Read the file content and append it to the output file
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(f"\n\n--- Content of {file_path} ---\n\n")
                            outfile.write(infile.read())
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Append all text files in a directory to an output file.")
    parser.add_argument('directory', help="Path to the directory containing text files.")
    parser.add_argument('output', help="Path to the output file where contents will be appended.")
    parser.add_argument('--extensions', nargs='*', default=None,
                        help="Optional list of file extensions to include (e.g., .txt .cpp). "
                             "Defaults to .txt, .cpp, .cu, .h, .py, .md if not specified.")

    # Parse arguments
    args = parser.parse_args()

    # Run the function with provided arguments
    append_text_files_to_output(args.directory, args.output, args.extensions)

if __name__ == "__main__":
    main()
