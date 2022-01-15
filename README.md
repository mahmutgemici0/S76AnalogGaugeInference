# Helicopter Cockpit Analog Gauge information Inference by using Deep Learning 

1. use save_videos_frame_from_path_file.py to process videos into frames.
2. find the timestamp and interested gauge's x1 x2 y1 y2 coordinates 
3. use crop_gauge_from_frame.py to crop the gauge and timestamp regions.
4. use Simdata_Tesseract_image_to_text.py to convert timestamp images into string and map it with ground truth csvs.
Now we have the gauge images and corresponding gauge values in final_csvs/ folder.
5. use train.py to train a model
