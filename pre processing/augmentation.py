import os 
import augly.video.functional as functional
import augly.video.composition as composition
import random
# function to obtain input and root output paths of videos 
def get_paths(video_folder, augmented_videos_folder):
    os.chdir(video_folder) 
    cwd = os.getcwd()
    directory = os.scandir()
    original_videos_list = []
    augmented_videos_list = []
    for file in directory:
        if file.name[-4 :] != '.mp4':
            continue 
        path = cwd + f'\{file.name}' 
        original_videos_list.append(path)
        output_path = augmented_videos_folder + f'\{file.name[-23: -4]}'
        augmented_videos_list.append(output_path)
    return original_videos_list, augmented_videos_list

# function to grayscale videos 
def perform_grayscale(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'grayscaled.mp4' 
        functional.grayscale(video_path=input_path, output_path=output_path)

# function to add gaussian noise 
def perform_add_noise(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'gaussian_noise.mp4' 
        functional.add_noise(video_path=input_path, output_path=output_path)

# function to blur the video 
def perform_blurring(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'blur.mp4' 
        functional.blur(video_path=input_path, output_path=output_path)

# function to change brightness of the video 
def change_brightness(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'brightness_changed.mp4' 
        functional.brightness(video_path=input_path, output_path=output_path, level=random.uniform(-1, 1))

# function to color jitter the video 
def color_jitter(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'color_jitter.mp4' 
        functional.color_jitter(video_path=input_path, output_path=output_path, brightness_factor=random.uniform(-1, 1), contrast_factor=random.uniform(-1000, 1000), saturation_factor= random.uniform(0, 3))

# function to horizontally flip the video 
def perform_horizontal_flip(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'hflip.mp4' 
        functional.hflip(video_path=input_path, output_path=output_path)

# overlay dots onto the video 
def overlay_dots(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'overlay_dots.mp4' 
        functional.overlay_dots(video_path=input_path, output_path=output_path, num_dots=500, dot_type='blur', random_movement=True)

# perform perspective transform 
def perform_perspective_transform_and_shake(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'perspective.mp4' 
        functional.perspective_transform_and_shake(video_path=input_path, output_path=output_path, shake_radius=5.0)
# rotate the video 
def perform_rotation(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'rotation.mp4' 
        functional.rotate(video_path=input_path, output_path=output_path, degrees=random.uniform(60, 270))

# scale the video resolution
def scale_resolution(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'resolution.mp4' 
        functional.scale(video_path=input_path, output_path=output_path, factor=random.uniform(0, 1))

# vertically flip the video 
def perform_vertical_flip(input_path_list, output_path_list):
    for index in range(len(input_path_list)): 
        input_path = input_path_list[index]
        output_path = output_path_list[index] + 'vflip.mp4' 
        functional.vflip(video_path=input_path, output_path=output_path)

# wrapper function to perform augmentations 
def perform_augmentations(input_path_list, output_path_list):
    # perform_grayscale(input_path_list, output_path_list)
    # perform_add_noise(input_path_list, output_path_list)
    # perform_blurring(input_path_list, output_path_list)
    # perform_horizontal_flip(input_path_list, output_path_list)
    # # perform_perspective_transform_and_shake(input_path_list, output_path_list)
    # perform_rotation(input_path_list, output_path_list)
    # perform_vertical_flip(input_path_list, output_path_list)
    # scale_resolution(input_path_list, output_path_list)
    # overlay_dots(input_path_list, output_path_list)
    color_jitter(input_path_list, output_path_list)
    change_brightness(input_path_list, output_path_list)
    



video_folder = r'D:\NITT\interns\iiit research summer intern\datasets\bp\bp recordings\BP Intern dataset'
augmented_videos_folder = r'D:\NITT\interns\iiit research summer intern\datasets\bp\bp recordings\rough videos'

# store input and output paths of all videos in a list

input_path_list, output_path_list = get_paths(video_folder, augmented_videos_folder)
perform_augmentations(input_path_list, output_path_list)
# print(output_path_list[1])


