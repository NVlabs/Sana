# @Author:  Pevernow (wzy3450354617@gmail.com)
# @Date:    2025/1/5
# @License: (Follow the main project)
import json
import os
import tarfile

from PIL import Image, PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # Increase maximum size for text chunks


def process_data(input_dir, output_tar_name="output.tar"):
    """
    Processes a directory containing PNG files, generates corresponding JSON files,
    and packages all files into a TAR file. It also counts the number of processed PNG images,
    and saves the height and width of each PNG file to the JSON.

    Args:
        input_dir (str): The input directory containing PNG files.
        output_tar_name (str): The name of the output TAR file (default is "output.tar").
    """
    png_count = 0
    json_files_created = []

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".png"):
            png_count += 1
            base_name = filename[:-4]  # Remove the ".png" extension
            txt_filename = os.path.join(input_dir, base_name + ".txt")
            json_filename = base_name + ".json"
            json_filepath = os.path.join(input_dir, json_filename)
            png_filepath = os.path.join(input_dir, filename)

            if os.path.exists(txt_filename):
                try:
                    # Get the dimensions of the PNG image
                    with Image.open(png_filepath) as img:
                        width, height = img.size

                    with open(txt_filename, encoding="utf-8") as f:
                        caption_content = f.read().strip()

                    data = {"file_name": filename, "prompt": caption_content, "width": width, "height": height}

                    with open(json_filepath, "w", encoding="utf-8") as outfile:
                        json.dump(data, outfile, indent=4, ensure_ascii=False)

                    print(f"Generated: {json_filename}")
                    json_files_created.append(json_filepath)

                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
            else:
                print(f"Warning: No corresponding TXT file found for {filename}.")

    # Create a TAR file and include all files
    with tarfile.open(output_tar_name, "w") as tar:
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            tar.add(item_path, arcname=item)  # arcname maintains the relative path of the file in the tar

    print(f"\nAll files have been packaged into: {output_tar_name}")
    print(f"Number of PNG images processed: {png_count}")


if __name__ == "__main__":
    input_directory = input("Please enter the directory path containing PNG and TXT files: ")
    output_tar_filename = (
        input("Please enter the name of the output TAR file (default is output.tar): ") or "output.tar"
    )
    process_data(input_directory, output_tar_filename)
