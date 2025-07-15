import torch
import os

import urllib.request

def download_checkpoint(url, save_path):
    """Download the SAM checkpoint from the given URL."""
    if not os.path.exists(save_path):
        print(f"Downloading checkpoint from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Checkpoint saved to {save_path}.")
    else:
        print(f"Checkpoint already exists at {save_path}.")

def remove_neck_from_checkpoint(checkpoint_path, output_path):
    """Load the checkpoint, remove the neck, and save the modified checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    #eliminate the head of the chekpoint
    model_state = {key: value for key, value in checkpoint.items() if "image_encoder" in key}
    # Assuming the neck is a specific part of the model's state_dict
    keys_to_remove = [key for key in model_state if "neck" in key]
    for key in keys_to_remove:
        print(f"Removing {key} from checkpoint...")
        del model_state[key]
    checkpoint = model_state
    # Remove 'image_encoder.' prefix from all keys
    checkpoint = {key.replace("image_encoder.", ""): value for key, value in checkpoint.items()}


    print(f"Saving modified checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)
    print("Modified checkpoint saved.")

if __name__ == "__main__":
    # URL to the SAM checkpoint
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    checkpoint_path = "sam_vit_l_image_encoder_orig.pth"
    modified_checkpoint_path = "sam_vit_l_image_encoder_no_neck.pth"

    # Step 1: Download the checkpoint
    download_checkpoint(checkpoint_url, checkpoint_path)

    # Step 2: Remove the neck and save the modified checkpoint
    remove_neck_from_checkpoint(checkpoint_path, modified_checkpoint_path)