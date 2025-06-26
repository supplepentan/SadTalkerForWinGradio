import os
import requests
import zipfile
from tqdm import tqdm


def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path, with progress bar."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

        with open(dest_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download.")
        else:
            print(f"Successfully downloaded {dest_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def unzip_file(zip_path, dest_dir):
    """Unzips a file to a destination directory."""
    if not os.path.exists(zip_path):
        print(f"Zip file not found: {zip_path}. Skipping unzip.")
        return

    print(f"Unzipping {zip_path} to {dest_dir}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # A simplified check to simulate -n (no-clobber) for directories.
            # If the destination directory exists and is not empty, assume already unzipped.
            # For more robust check, one might iterate zip_ref.namelist() and check each file.
            if os.path.exists(dest_dir) and os.listdir(dest_dir):
                print(
                    f"Destination directory {dest_dir} is not empty. Assuming already unzipped. Skipping."
                )
                return

            os.makedirs(dest_dir, exist_ok=True)
            zip_ref.extractall(dest_dir)
        print(f"Successfully unzipped {zip_path}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred during unzipping: {e}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    gfpgan_weights_dir = os.path.join(base_dir, "gfpgan", "weights")

    # Create directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(gfpgan_weights_dir, exist_ok=True)
    print(f"Created directory: {checkpoints_dir}")
    print(f"Created directory: {gfpgan_weights_dir}")

    # Define files to download (URL, destination_filename, destination_directory)
    files_to_download = [
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
            "mapping_00109-model.pth.tar",
            checkpoints_dir,
        ),
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
            "mapping_00229-model.pth.tar",
            checkpoints_dir,
        ),
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
            "SadTalker_V0.0.2_256.safetensors",
            checkpoints_dir,
        ),
        (
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
            "SadTalker_V0.0.2_512.safetensors",
            checkpoints_dir,
        ),
        (
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
            "alignment_WFLW_4HG.pth",
            gfpgan_weights_dir,
        ),
        (
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "detection_Resnet50_Final.pth",
            gfpgan_weights_dir,
        ),
        (
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "GFPGANv1.4.pth",
            gfpgan_weights_dir,
        ),
        (
            "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "parsing_parsenet.pth",
            gfpgan_weights_dir,
        ),
    ]

    for url, filename, dest_dir in files_to_download:
        download_file(url, os.path.join(dest_dir, filename))

    # The original shell script had commented-out sections for hub.zip and BFM_Fitting.zip.
    # If these were to be included, uncomment and adjust the following:
    # hub_zip_path = os.path.join(checkpoints_dir, "hub.zip")
    # download_file("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/hub.zip", hub_zip_path)
    # unzip_file(hub_zip_path, checkpoints_dir)

    # bfm_zip_path = os.path.join(checkpoints_dir, "BFM_Fitting.zip")
    # download_file("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip", bfm_zip_path)
    # unzip_file(bfm_zip_path, checkpoints_dir)

    print("\nAll specified models and weights processed.")


if __name__ == "__main__":
    main()
