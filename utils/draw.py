import cv2
import os
import torch
import numpy as np

from ultralytics.engine.results import Results

def get_text_color(box_color):
    text_color = (255,255,255)

    brightness = box_color[2]*0.299 + box_color[1]*0.587 + box_color[0]*0.114

    if(brightness > 180):
        text_color = (0, 0, 0)

    return text_color

def box(img, detection_output, class_list, colors) :    
    # Copy image, in case that we need original image for something
    out_image = img 

    for run_output in detection_output :
        # Unpack
        label, con, box = run_output        

        # Choose color
        box_color = colors[int(label.item())]
        # text_color = (255,255,255)
        text_color = get_text_color(box_color)
        # Get Class Name
        label = class_list[int(label.item())]
        # Draw object box
        first_half_box = (int(box[0].item()),int(box[1].item()))
        second_half_box = (int(box[2].item()),int(box[3].item()))
        cv2.rectangle(out_image, first_half_box, second_half_box, box_color, 2)
        # Create text
        text_print = '{label} {con:.2f}'.format(label = label, con = con.item())
        # Locate text position
        text_location = (int(box[0]), int(box[1] - 10 ))
        # Get size and baseline
        labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 1) 
        
        # Draw text's background
        cv2.rectangle(out_image 
                        , (int(box[0]), int(box[1] - labelSize[1] - 10 ))
                        , (int(box[0])+labelSize[0], int(box[1] + baseLine-10))
                        , box_color , cv2.FILLED)        
        # Put text
        cv2.putText(out_image, text_print ,text_location
                    , cv2.FONT_HERSHEY_SIMPLEX , 1
                    , text_color, 2, cv2.LINE_AA)

    return out_image

def fps(avg_fps, combined_img):        
    avg_fps_str = float("{:.2f}".format(avg_fps))
    
    cv2.putText(combined_img, "FPS: "+str(avg_fps_str), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return combined_img

