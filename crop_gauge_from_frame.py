#USAGE
# python parser.py -video_path test.mp4 -config config.csv -gauge_name timestamp


# import the necessary packages
import argparse
import pandas as pd
import os
import shutil
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True,
    help="the sim video path in broomcloset_images folder")
ap.add_argument("-config", "--config", required = True,
	help="global variables")
ap.add_argument("-gauge_name", "--gauge_name", required = True,
	help="Name of the gauge that is going to be cropped!")

args = ap.parse_args()

#global variables
VIDEO_NAME = args.video_path
CONFIG = args.config
GAUGE_NAME = args.gauge_name
VIDEO_PATH = "broomcloset_images/{}".format(VIDEO_NAME)

#read config file
CONFIG = pd.read_csv('{}'.format(CONFIG), index_col = 'video_name')

#get x,y coordinates indexes
idx = CONFIG.index
video_name_index = idx.get_loc('{}'.format(VIDEO_NAME))

#find coordinates
gauge_x1 = CONFIG['{}_x1'.format(GAUGE_NAME)][video_name_index]
gauge_y1 = CONFIG['{}_y1'.format(GAUGE_NAME)][video_name_index]
gauge_x2 = CONFIG['{}_x2'.format(GAUGE_NAME)][video_name_index]
gauge_y2 = CONFIG['{}_y2'.format(GAUGE_NAME)][video_name_index]

print("{}'s (x,y) coordinates are: {},{},{},{}".format(GAUGE_NAME,gauge_x1,gauge_y1, gauge_x2, gauge_y2))

def crop_gauge_from_frame(path, gauge_name, x1, y1, x2, y2):
    
    parent_dir, vid_name = os.path.split(path)
    gauge_images = 'gauge_images'
    
    if os.path.exists('{0}/{1}/{2}'.format(gauge_images, gauge_name, vid_name)):
    	print('DELETING EXISTED DIRECTORY...')
    	shutil.rmtree('{0}/{1}/{2}'.format(gauge_images, gauge_name, vid_name))
        
    if not os.path.exists('{0}/{1}/{2}'.format(gauge_images, gauge_name, vid_name)):
      # Create a new directory because it does not exist 
        os.makedirs('{0}/{1}/{2}'.format(gauge_images, gauge_name,vid_name))
        print("{0}/{1}/{2} is created.".format(gauge_images,gauge_name,vid_name))
    else:
        print("{0}/{1}/{2} is already exists.".format(gauge_images,gauge_name,vid_name))
        
    dirs = os.listdir(path) 
    print('CROPPING REGION OF INTEREST...')   
    for item in dirs:
        if item == ".DS_Store":          #MacOS thing
            continue
            
        fullpath = os.path.join(path,item)         #corrected
#         print(fullpath)
    
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)  
            imCrop = im.crop((x1, y1, x2, y2)) #corrected
            head,tail = os.path.split(fullpath)
            parent_path, vid_names = os.path.split(head)
            imCrop.save('{0}/{1}/{2}/{3}'.format(gauge_images, gauge_name, vid_names, tail))


crop_gauge_from_frame(VIDEO_PATH,
                      GAUGE_NAME,
                       x1 = gauge_x1, y1 = gauge_y1, x2 = gauge_x2, y2 = gauge_y2)