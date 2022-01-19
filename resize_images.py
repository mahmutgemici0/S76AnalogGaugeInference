import PIL
import os
import os.path
from PIL import Image
import timeit

def resize(img_dir, w, h):
    print('Bulk images resizing started...')

    for img in os.listdir(img_dir):
        f_img = img_dir + img
        f, e = os.path.splitext(img_dir + img)
        img = Image.open(f_img)
        img = img.resize((w, h))
        img.save(f + '.jpeg')

    print('Bulk images resizing finished...')

resize(test_image_folder_path, 67,67)