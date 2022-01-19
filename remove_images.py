import os
import glob
import time
def remove_images(csv, folder):
    image_path_list = []

    for i in csv['image_path']:
        image_path_list.append(i)

    for i in os.listdir(folder):
        if i not in image_path_list:
            os.remove(folder + i)

remove_images(test_csv, test_image_folder_path)