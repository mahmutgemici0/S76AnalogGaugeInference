# usage : Simdata_Tesseract_image_to_text.py --video_name "2022-01-01 17-31-30_Trim1.mp4" --csv_path CSVs/SimData_2022.01.01_17.31.22.txt.csv
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import cv2
import multiprocessing
import os
import sys
import time
import pandas as pd
from PIL import Image
import pytesseract
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# %%
from os import listdir
from os.path import isfile, join
import numpy
import shutil
import argparse


# DEFINE ARGUMENTS
ap = argparse.ArgumentParser()
ap.add_argument("-video_name", "--video_name", required=True,
    help="name of the video")
ap.add_argument("-csv_path", "--csv_path", required=True,
    help="path of the sim data csv")


args = ap.parse_args()

#arg variables
VIDEO_NAME = args.video_name
CSV_PATH = args.csv_path


# video_name = input('Please enter video path: ')
path = 'gauge_images/timestamp/{}/'.format(VIDEO_NAME)
images = [ f for f in listdir(path) if isfile(join(path,f)) ]
df_sim = pd.read_csv('{}'.format(CSV_PATH))
# df_sim = pd.read_csv('CSVs/{}'.format(input('Please enter CSV path: ')))

# %%
# import re

# def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
#     return [
#         int(text)
#         if text.isdigit() else text.lower()
#         for text in _nsre.split(s)]

# sorted_images = sorted(listIm, key=natural_sort_key)

# %%
from PIL import Image

def get_ocr1(path):
    text=[]
    image=[]
    dirs = os.listdir(path)
    for crop in dirs:
        if crop =='.DS_Store':
            pass
        else:
            tmp=pytesseract.image_to_string(Image.open(path + crop))
            text.append(tmp)
            image.append(crop)
#             time.sleep(0.01)
    return text,image


def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        future1 = executor.submit(get_ocr1, path) #1. video


        
    my_ocrs1 = future1.result()

#     my_ocrs5 = future5.result()

    return my_ocrs1

# %%
print('Getting timestamp from images...')
start = time.time()

timetext = main()

end = time.time()
print('DONE! \nIt took:' + str(end-start) + 'seconds')

# %%
timestamp = []
image_path = []
for i in timetext[0]:
    i = i[0:10]
    timestamp.append(i)

for i in timetext[1]:
    image_path.append(i)

print('First timestamp :{}\nFirst image:{}'.format(timestamp[0], image_path[0]))

# %%
df_ocr = pd.DataFrame({"System UTC Time": timestamp,
                        'image_path': image_path})

print('First row:{}'.format(df_ocr.head(1)))

# Map 2 dataframes on column values
def inner_merge(df_gt, df_ocr):
    df = pd.merge(df_gt, df_ocr, how = 'inner', on = 'System UTC Time')
    return df

# %%
# select first 10 digits of timestamp
def trim_timestamp(df = df_sim['System UTC Time']):
    for i in df:
        j = i[0:10]
        df = df.replace([i],j)
    return df

# %%
# 
for i in df_sim['System UTC Time']:
    j = i[0:10]
    df_sim['System UTC Time'] = df_sim['System UTC Time'].replace([i],j)

# %%
#trim timestamp 
# df_sim = trim_timestamp(df = df_sim['System UTC Time'])

#map csvs
merged_df = inner_merge(df_sim, df_ocr)

#convert float to integer (labels)
merged_df['Airspeed(Ind)'] = pd.to_numeric(merged_df['Airspeed(Ind)']).astype(int)

#save as csv
merged_df.to_csv('train_{}.csv'.format(VIDEO_NAME))
print('TRAINING CSV FILE HAS CREATED!!\n{}'.format(merged_df.head(1)))