import cv2
import os
import argparse

default_output_folder = "/home/pettepiero/tirocinio/dial-a-ride/outputs/videos"
desidered_fps = 12

parser = argparse.ArgumentParser(description="Generate video from images")
parser.add_argument(
    "--image_folder",
    type=str,
    default=None,
    help="Folder containing images to be converted to video",
)
parser.add_argument(
    "--video_name",
    type=str,
    default="video.avi",
    help="Name of the video file to be generated",
)

args = parser.parse_args()

if args.image_folder is None:
    print("Please provide the path to the folder containing images")
    exit()

if not os.path.exists(default_output_folder):
    os.makedirs(default_output_folder)

image_folder = args.image_folder
video_name = f"{default_output_folder}/{args.video_name}"

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, desidered_fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
