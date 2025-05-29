import os
import zipfile
import argparse

def unzip_data(zip_path: str, output_dir: str = "data/") -> str:
    """
    Unzips a zip file to the specified directory.

    Args:
        zip_path (str): Path to the local zip file.
        output_dir (str): Directory where the zip content will be extracted.

    Returns:
        str: Path to the extracted data folder.
    """

    zip_path = os.path.abspath(zip_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found at {zip_path}")
    
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        if not namelist:
            raise ValueError("Zip file is empty.")

        first_file_path = os.path.join(output_dir, namelist[0])
        if os.path.exists(first_file_path):
            print(f"Data already extracted to {output_dir}. Skipping unzip.")
        else:
            print(f"Extracting {zip_path} to {output_dir}...")
            zip_ref.extractall(output_dir)
            print(f"Extracted {len(namelist)} files to {output_dir}.")

    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip a dataset into a target folder.")
    parser.add_argument("zip_path", type=str, help="Path to the local zip file")
    parser.add_argument("--output_dir", type=str, default="data/", help="Directory to extract the contents (default: data/)")

    args = parser.parse_args()

    extracted_path = unzip_data(args.zip_path, args.output_dir)
    print(f"Data is available at: {extracted_path}")