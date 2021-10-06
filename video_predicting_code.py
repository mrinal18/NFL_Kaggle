#predicting helmets from video of NFL using object detection

import numpy as np
import pandas as pd
import itertools
import glob
import os
import cv2
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import random


class HelmetDetector:
    def __init__(self, model_path, labels_path, num_clusters=5, num_workers=4):
        self.model_path = model_path
        self.labels_path = labels_path
        self.num_clusters = num_clusters
        self.num_workers = num_workers
        self.model = cv2.dnn.readNetFromTensorflow(self.model_path)
        self.labels = self.load_labels(self.labels_path)
        self.cluster_centers = None
        self.cluster_labels = None

    def load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = self.labels[class_id]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def predict(self, img):
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        output_layers = self.get_output_layers(self.model)
        layer_outputs = self.model.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    width = int(detection[2] * img.shape[1])
                    height = int(detection[3] * img.shape[0])
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_prediction(img, class_ids[i], confidences[i], left, top, left + width, top + height)

    def cluster_data(self, data):
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(data)
        self.cluster_centers = kmeans.cluster_centers_
        self.cluster_labels = kmeans.labels_

    
    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.predict(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    # Modified function from to take single frame.
# https://www.kaggle.com/samhuddleston/nfl-1st-and-future-getting-started

# https://www.kaggle.com/nvnnghia/evaluation-metrics
from scipy.optimize import linear_sum_assignment

def iou_ltwh(bbox1, bbox2):
    
    
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    
    bbox1[2] += bbox1[0] 
    bbox1[3] += bbox1[1] 
    
    bbox2[2] += bbox2[0] 
    bbox2[3] += bbox2[1] 

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def precision_calc(gt_boxes, pred_boxes):
    cost_matix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0]-box2[0])
            if dist > 4:
                continue
            iou_score2 = iou_ltwh(box1[1:], box2[1:])

            if iou_score2 < 0.35:
                continue
            else:
                cost_matix[i,j]=0

    row_ind, col_ind = linear_sum_assignment(cost_matix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp=0
    for i, j in zip(row_ind, col_ind):
        if cost_matix[i,j]==0:
            tp+=1
        else:
            fp+=1
            fn+=1
    return tp, fp, fn

def competition_metric(valid_labels, pred_df, output=False):
    ftp, ffp, ffn = [], [], []
    cols = ['frame', 'left', 'top', 'width', 'height']
    for video in valid_labels['video'].unique():
        pred_boxes = pred_df[pred_df['video'] == video][cols].values
        gt_boxes = valid_labels[valid_labels['video'] == video][cols].values
       
        tp, fp, fn = precision_calc(gt_boxes, pred_boxes)
        ftp.append(tp)
        ffp.append(fp)
        ffn.append(fn)
    
    tp = np.sum(ftp)
    fp = np.sum(ffp)
    fn = np.sum(ffn)
    precision = tp / (tp + fp + 1e-6)
    recall =  tp / (tp + fn +1e-6)
    f1_score = 2*(precision*recall)/(precision+recall+1e-6)
    if output:
        return tp, fp, fn, precision, recall, f1_score
    else:
        print(f'TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {precision:.4f}, RECALL: {recall:.4f}, F1 SCORE: {f1_score}')

true = train.loc[(train.video.isin(valid_videos))&(train.impact==1)&(train.confidence>1)&(train.visibility>0)]
print('There are %i ground truths for the videos in OOF file'%true.shape[0] )
print()
print('This OOF file has competition metric:')
competition_metric(true, df_pred)


def annotate_frame(gameKey, playID, video_labels, slow=1, stop_frame=-1, start_frame=-1) -> str:
    VIDEO_CODEC = "MP4V"
    BLACK = (0, 0, 0)    # Black
    WHITE = (255, 255, 255)    # White
    IMPACT_COLOR = (0, 0, 255)  # Red
    PRED_COLOR = (255, 0, 0) # Blue
    PRED_COLOR_WARN1 = (0, 255, 255) # Yellow
    PRED_COLOR_WARN2 = (0, 255, 0) # Green
    
    tp, fp, fn, pp, rr, ff = competition_metric(true.loc[(true.gameKey==gameKey)&(true.playID==playID)],
                               df_pred.loc[(df_pred.gameKey==gameKey)&(df_pred.playID==playID)],True)
    
    video_path1 = BASE_DIR+'/train/%i_%.6i_Endzone.mp4'%(gameKey,playID)
    video_path2 = BASE_DIR+'/train/%i_%.6i_Sideline.mp4'%(gameKey,playID)
    
    video_name1 = os.path.basename(video_path1)
    video_name2 = os.path.basename(video_path2)
    
    hits1 = train.loc[train.video==video_name1].drop_duplicates('frame').sort_values('frame').hit.values
    f_max1 = train.loc[train.video==video_name1,'frame'].max()
    hits2 = train.loc[train.video==video_name2].drop_duplicates('frame').sort_values('frame').hit.values
    f_max2 = train.loc[train.video==video_name2,'frame'].max()
    
    hits3 = df_pred.loc[df_pred.video==video_name1].frame.unique()
    hits4 = df_pred.loc[df_pred.video==video_name2].frame.unique()
    
    if f_max1 != f_max2:
        print('## WARNING: different length videos')
    f_max = min(f_max1,f_max2)
    print('Converting',f_max,'frames...',end='')
    
    vidcap1 = cv2.VideoCapture(video_path1)
    vidcap2 = cv2.VideoCapture(video_path2)
    
    fps = vidcap1.get(cv2.CAP_PROP_FPS)
    width1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "labeled_" + video_name1.replace('_Endzone','')
    tmp_output_path = "tmp_" + output_path
    output_video = cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), fps/slow, (width1, height1))
    
    frame = 0
    while True:
        
        if frame%10==0: print(frame,', ',end='')
        img = np.zeros((height1,width1,3),dtype='uint8')
                
        it_worked1, img1 = vidcap1.read()
        if not it_worked1: break
            
        it_worked2, img2 = vidcap2.read()
        if not it_worked2: break
            
        if frame<start_frame: 
            frame += 1
            continue
        if frame==stop_frame: break
            
        img[360:,:640,:] = img1[::2,::2,:]
        img[360:,640:,:] = img2[::2,::2,:]
        
        # We need to add 1 to the frame count to match the label frame index that starts at 1
        frame += 1
        
        # Let's add a frame index to the video so we can track where we are
        img_name = f"GamePlay_{video_name1.replace('_Endzone.mp4','')}_frame{frame}"
        cv2.putText(img, img_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, thickness=2)
        
        metric = f'TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {pp:.3f}, RECALL: {rr:.3f}, F1 SCORE: {ff:.4f}'
        cv2.putText(img, metric, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, thickness=1)
            
        # MAKE FOUR PROGRESS LINES
        hh = 100
        cv2.line(img, (20,hh),(600,hh),(0,0,255),4)
        for k in np.where( hits1==1 )[0]:
            x = int(k/f_max * 580 + 20)
            cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(0,0,255),cv2.FILLED) 
        x = int(frame/f_max * 580 + 20)
        cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(255,255,255),cv2.FILLED) 
        
        hh = 150
        cv2.line(img, (20,hh),(600,hh),(0,0,255),4)
        for k in np.where( hits2==1 )[0]:
            x = int(k/f_max * 580 + 20)
            cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(0,0,255),cv2.FILLED) 
        x = int(frame/f_max * 580 + 20)
        cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(255,255,255),cv2.FILLED) 
        
        hh = 200
        cv2.line(img, (20,hh),(600,hh),(255,0,0),4)
        for k in hits3:
            x = int(k/f_max * 580 + 20)
            cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(255,0,0),cv2.FILLED) 
        x = int(frame/f_max * 580 + 20)
        cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(255,255,255),cv2.FILLED) 
        
        hh = 250
        cv2.line(img, (20,hh),(600,hh),(255,0,0),4)
        for k in hits4:
            x = int(k/f_max * 580 + 20)
            cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(255,0,0),cv2.FILLED) 
        x = int(frame/f_max * 580 + 20)
        cv2.rectangle(img, (x-1,hh-10),(x+1,hh+10),(255,255,255),cv2.FILLED) 
        
        w1, w2, h1 = [], [], []
        
        # DRAW 4 SETS OF BOXES
        boxes = video_labels.query("video == @video_name1 and frame == @frame and warning != 0")
        for box in boxes.itertuples(index=False):
            left = box.left//2
            top = box.top//2 + 360
            width = box.width//2
            height = box.height//2
            if box.impact == 1 and box.confidence > 1 and box.visibility > 0:   
                color, thickness = IMPACT_COLOR, 2
                #print('(Impact frame',frame,box.label,box.confidence,box.visibility,')',end='')  
                h1.append(box.label)
            elif box.warning == 1:    
                color, thickness = WHITE, 1
                w1.append(box.label)
            else:
                color, thickness = BLACK, 1
                w2.append(box.label)
            # Add a box around the helmet
            cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness=thickness)
            #cv2.putText(img, box.label, (left, max(0, top - 5//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
            
        # Now, add the boxes
        boxes = video_labels.query("video == @video_name2 and frame == @frame and warning != 0")
        for box in boxes.itertuples(index=False):
            left = box.left//2 + 640
            top = box.top//2 + 360
            width = box.width//2
            height = box.height//2
            if box.impact == 1 and box.confidence > 1 and box.visibility > 0:   
                color, thickness = IMPACT_COLOR, 2
                #print('Impact frame',frame,box.label,box.confidence,box.visibility)            
            elif box.warning == 1:    
                color, thickness = WHITE, 1
            else:
                color, thickness = BLACK, 1
            # Add a box around the helmet
            cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness=thickness)
            #cv2.putText(img, box.label, (left, max(0, top - 5//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
            
            
        #Now, add the boxes
        boxes = df_pred.loc[(df_pred.video == video_name1) & (abs(df_pred.frame - frame)<=10)]
        for box in boxes.itertuples(index=False):
            left = box.left//2
            top = box.top//2 + 360
            width = box.width//2
            height = box.height//2
            if box.frame == frame:   
                color, thickness = PRED_COLOR, 2
                #print('(Pred frame',frame,')',end='')  
            elif box.frame > frame:
                color, thickness = PRED_COLOR_WARN1, 1
            else:
                color, thickness = PRED_COLOR_WARN2, 1
                
            # Add a box around the helmet
            cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness=thickness)
            #cv2.putText(img, box.label, (left, max(0, top - 5//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
                              
        #Now, add the boxes
        boxes = df_pred.loc[(df_pred.video == video_name2) & (abs(df_pred.frame - frame)<=10)]
        for box in boxes.itertuples(index=False):
            left = box.left//2 + 640
            top = box.top//2 + 360
            width = box.width//2
            height = box.height//2
            if box.frame == frame:   
                color, thickness = PRED_COLOR, 2
                #print('(Pred frame',frame,')',end='') 
            elif box.frame > frame:
                color, thickness = PRED_COLOR_WARN1, 1
            else:
                color, thickness = PRED_COLOR_WARN2, 1

            # Add a box around the helmet
            cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness=thickness)
            #cv2.putText(img, box.label, (left, max(0, top - 5//2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=1)
            
        # DRAW ARIAL VIEW WITH TRACKING INFO
        img3 = get_track_image(gameKey,playID,fps,frame+1,f_max, warn1=w1, warn2=w2, hit=h1)
        img[:360,640:,:] = img3[::2,::2,:]    
        
        
        output_video.write(img)
    output_video.release()
    
    # Not all browsers support the codec, we will re-load the file at tmp_output_path and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(output_path):
        os.remove(output_path)
    subprocess.run(["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast", "-vcodec", "libx264", output_path])
    os.remove(tmp_output_path)
    
    return output_path
    
    

