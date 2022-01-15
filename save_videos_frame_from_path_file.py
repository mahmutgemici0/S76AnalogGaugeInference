import cv2
import os
import shutil
import argparse

# DEFINE ARGUMENTS
ap = argparse.ArgumentParser()
ap.add_argument("-freq", "--freq", required=True,
    help="How often do you want to extract frames?")

args = ap.parse_args()

#arg variables
FREQUENCY = args.freq

# list videofiles
directory = 'broomcloset_videos'
directories = os.listdir(directory)

# convert list(videofiles) into txt file
with open('video_path_names.txt', 'w') as f:
    for path in directories:
        f.write('{}/{}'.format(directory,path))
        f.write('\n')
          

def save_videos_frame_from_path_file(video_path = "./VideosPaths", output_path = "./VideosSampleFrames", freq = 25):
    
    #if broomcloset_image is already existed it will be deleted. 
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Storing frames path:" + output_path)
                
                
    with open(video_path, 'r') as f:
        paths = f.read().split("\n")
    #     print(len(content))
#         print(paths)
        for v_path in paths:
            print(v_path)
            video_name = v_path.split("/")[-1]
            video_frame_path = "{0}/{1}".format(output_path, video_name)
            if not os.path.exists(video_frame_path):
                os.makedirs(video_frame_path)
#                 print("Video name=", video_frame_path)


                vcap = cv2.VideoCapture(v_path)
                if not vcap.isOpened():
                    exit(0)

                #Capture images per 25 frame
                frameFrequency = freq

                #iterate all frames
                total_frame = 0
                id = 0
                while True:
                    ret, frame = vcap.read()
                    if ret is False:
                        break
                    total_frame += 1
                    if total_frame%frameFrequency == 0:
                        id += 1
#                         image_name = video_images + str(id) +'.jpg'
                        cv2.imwrite("{0}/{1}_{2}.jpeg".format(video_frame_path, id, video_name), frame)
#                         cv2.imwrite(image_name, frame)
#                         print(image_name)

                vcap.release()

# %%
save_videos_frame_from_path_file(video_path = "video_path_names.txt", output_path = "./broomcloset_images" , freq = int(FREQUENCY))

# %%
