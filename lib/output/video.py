import os
from datetime import datetime
import cv2


def generate_video(
    image_base_folder: str = "./outputs/plots",
    default_output_folder: str = "./outputs/videos",
    desidered_fps: int = 12,
):
    """
    Generates a video from images stored in the most recent folder inside `image_base_folder`.

    Parameters:
    - image_base_folder (str): The parent folder containing timestamped image subfolders.
    - default_output_folder (str): The folder where the output video will be saved.
    - desidered_fps (int): The frames per second for the output video.
    """

    # Ensure the base plots folder exists
    if not os.path.exists(image_base_folder):
        print(f"Folder {image_base_folder} not found.")
        return

    # Get all subdirectories sorted by name (assuming they are timestamped)
    subfolders = sorted(
        [
            f
            for f in os.listdir(image_base_folder)
            if os.path.isdir(os.path.join(image_base_folder, f))
        ],
        reverse=True,  # Latest folder first
    )

    if not subfolders:
        print(f"No subfolders found in {image_base_folder}.")
        return

    # Select the most recent folder
    latest_folder = os.path.join(image_base_folder, subfolders[0])
    print(f"Using images from: {latest_folder}")

    # Ensure output folder exists
    os.makedirs(default_output_folder, exist_ok=True)

    # Generate video name with current timestamp
    video_name = datetime.now().strftime("%Y%m%d%H%M")
    file_name = os.path.join(default_output_folder, f"{video_name}.avi")

    # Get sorted images from the latest folder
    images = sorted([img for img in os.listdir(latest_folder) if img.endswith(".png")])

    if not images:
        print(f"No images found in {latest_folder}.")
        return

    # Read first image to get dimensions
    frame = cv2.imread(os.path.join(latest_folder, images[0]))
    height, width, layers = frame.shape

    # Create video writer
    video = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc(*"XVID"), desidered_fps, (width, height)
    )

    # Write images to video
    for image in images:
        video.write(cv2.imread(os.path.join(latest_folder, image)))

    # Release video writer
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved at: {file_name}")
