#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dazmer
"""
# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import pickle
import cv2
import time
import datetime
from os import path

def IoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def centroid_distances(old_rect_list, new_rect_list, key):
    olderCentroids = [key(c) for c in old_rect_list]
    inputCentroids = [key(c) for c in new_rect_list]
    D = dist.cdist(np.array(olderCentroids), inputCentroids)
    return D

def overlap_ratio(old_rect_list, new_rect_list, key):
    D = []
    for old_rect in old_rect_list:
        current_d = []
        for new_rect in new_rect_list:
            current_d.append(1-IoU(old_rect,new_rect))
        D.append(np.array(current_d))
    D = np.array(D)
    return D
     
def check_img_diff(obj, old_img, new_img, obtain_bounding):
    if(old_img is not None):
        # total_probability = 0
        # count = 0
        # for det in iterize(label_prev):
            # if det.has_key("bbox"):
        old_patch = obtain_bounding(obj, old_img)
        new_patch = obtain_bounding(obj, new_img)
        try:
            match = cv2.matchTemplate(new_patch, old_patch,
                                      cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(match)
        except:
            confidence=0
        # total_probability+=confidence
#            cv2.imshow("prev_img", det["img"])
#            cv2.imshow("new__img",  img_patch)
#            print "RES:", confidence
        # count += 1
#            cv2.waitKey(0)
        # total_probability /= float(count)
        return confidence
    else:
        return 1

class MultiTracker():
    def __init__(self, ttype, maxDisappeared=50, key=lambda x:x, get_patch=None,
                 update_obj = lambda old, new: new, pix_tol=500,
                 iou_threshold=0,
                 status_save=False, status_time_recover=("00:30:00"),
                 status_fpath="track_status{}.pkl"):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.object_paths = OrderedDict()
        self.iou_threshold = iou_threshold
        self.ttype = ttype
        
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
  
        self.key = key
        self.get_patch = get_patch
        self.update_obj = update_obj
        self.pixel_tolerance = pix_tol
        
        self.prev_img = None
        
        self.status_save = status_save
        x = time.strptime(status_time_recover,'%H:%M:%S')
        sx = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        self.status_time_recover=sx
        self.status_fpath=status_fpath
    def save_status(self):
        with open(self.status_fpath, "wb") as f:
            now_time = time.time()
            pickle.dump([now_time, self.nextObjectID, self.objects, self.disappeared], f)
    def recover_status(self):
        if(path.exists(self.status_fpath)):
            with open(self.status_fpath, "rb") as f:
                old_time, n_objs, objects, disap = pickle.load(f)
                time_elaps = (time.time() - old_time)
                has_to_rec = time_elaps < self.status_time_recover
                if(has_to_rec):
                    self.objects = objects  
                    self.nextObjectID = n_objs
                    self.disappeared = disap
    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.object_paths[self.nextObjectID] = [self.key(centroid)]
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def toggle_disappeared(self, objectID, new_img=None):
        self.disappeared[objectID] += 1
        
        obj = self.objects[objectID]
        if new_img is not None:
            prob = check_img_diff(obj, self.prev_img, new_img, self.get_patch)
        else:
            prob = 1.0
        if(prob < 0.9 and self.disappeared[objectID] < self.maxDisappeared):
            self.disappeared[objectID] = self.maxDisappeared
        # if we have reached a maximum number of consecutive
        # frames where a given object has been marked as
        # missing, deregister it
        if self.disappeared[objectID] > self.maxDisappeared:
            self.deregister(objectID)
            
    def update(self, rects, new_img=None):
        # if(new_img is not None and obtain_bounding)
        # check to see if the list of input bounding box rectangles
        # is empty
        if self.status_save and self.nextObjectID == 0:
            self.recover_status()
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared

            for objectID in self.disappeared.copy().keys():
                self.toggle_disappeared(objectID, new_img)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="float")

        # loop over the bounding box rectangles
        for (i, obj) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
#            cX = int((startX + endX) / 2.0)
#            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = self.key(obj)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(rects[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            # objectCentroids = [self.key(c) for c in list(self.objects.values())]

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            
            ### COMPUTING DISTANCES ###
            centroid_dist = centroid_distances(self.objects.values(), rects, self.key)
            if(self.ttype == "centroid"):
                # D = dist.cdist(np.array(objectCentroids), inputCentroids)
                D = centroid_dist
            elif(self.ttype == "overlap"):
                D = overlap_ratio(self.objects.values(), rects, self.key)
            else:
                raise(ValueError(f"ttype not recognized: {self.ttype}"))
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                if self.ttype == "overlap" and D[row][col] >= (1-self.iou_threshold):
                    continue
                if centroid_dist[row][col] > self.pixel_tolerance:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                # self.objects[objectID] = rects[col]
                old = self.objects[objectID]
                self.objects[objectID]= self.update_obj(old, rects[col])
                del old
                self.disappeared[objectID] = 0
                self.object_paths[objectID].append(self.key(rects[col]))

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.toggle_disappeared(objectID, new_img)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(rects[col])

        # return the set of trackable objects
        self.prev_img = new_img
        if(self.status_save): 
            self.save_status()
        return self.objects