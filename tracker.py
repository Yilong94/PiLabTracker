#from flask import Flask, render_template
#from flask_socketio import SocketIO, emit
#from threading import Lock 
#from database import getLogCSV, pushLogCSV

from datetime import date, datetime, timedelta
import time
import threading
import argparse
import imutils
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np
import math 
from database.DatabaseConnector import DatabaseConnector

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.

async_mode = None

#app = Flask(__name__)
#socketio = SocketIO(app, async_mode = async_mode)
#thread = None
#thread_lock = Lock()

databaseFilepath = 'database/record.csv'
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
host = 'pilab.ct0oc3ontoyk.ap-southeast-1.rds.amazonaws.com'
database = 'pilab'
csvfile = "database/log.csv"
db_connector = DatabaseConnector(username, password, host, database, csvfile)

#functionality 1: continuously send data to client on current count
# def background_thread():
#     print ('background working')

#     # count = 5

#     while True:

#         socketio.sleep(1)

#         #constantly update curent count in pilab
#         socketio.emit('currentCount', count, namespace = '/pilab')

#         #constantly update live graph
#         data = getLogCSV.getLiveCount_linechart(databaseFilepath)
#         socketio.emit('liveGraph', data, namespace = '/pilab')


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
     
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def humancounter():
    global count
    count = 0

    fgbg = cv2.createBackgroundSubtractorMOG2()
        #fgbg.setDetectShadows = False

    global camera
    camera  = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.rotation = 180
    global rawCapture 
    rawCapture = PiRGBArray(camera, size=(640, 480))
    time.sleep(0.5)
    global cam_prev_count 
    cam_prev_count = 0
    global prev_point 
    prev_point= []

    global distance_threshold 
    distance_threshold= 20
    global prevcount 
    prevcount = 0
    global enter 
    enter = 0
    global exit 
    exit = 0
    global update 
    update= False
    # initialize the first frame in the video stream
    global firstFrame 
    firstFrame= None
    # loop over the frames of the video
    
    #initialize time tracker for loading of data
    inputtime = '-1'

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array
        text = "Unoccupied"
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        frameDelta = fgbg.apply(frame)

        thresh = cv2.threshold(frameDelta, 245, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        _box = []
        for c in cnts:
            c = cv2.convexHull(c)
            if cv2.contourArea(c) < 8000:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            if (float(w)/float(h)>2 or float(h)/float(w)>2):
                continue
            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h
            r=(x1, x2, y1, y2)
            _box.append(r)
        
        _box = np.array(_box)
        _point= []
        _box = non_max_suppression_fast(_box, 0.01)
        for c in _box:
            cv2.rectangle(frame, (c[0], c[2]), (c[1], c[3]), (0, 255, 0), 2)
            p = (int((c[0]+c[1])/2), int((c[2]+c[3])/2))
            cv2.circle(frame, (p[0],  p[1]), 10, (0, 0, 255), 2) 
            _point.append(p)
        text = len(_box)
        ## determine person enter or exit 
        in_frame = len(_point)
        if in_frame > cam_prev_count:
            extra = []
            for p in _point:
                #check if point is any of prev point, if it is, stop checking, move to nets point 
                exist = False    
                for pp in prev_point:
                    if(math.hypot(pp[0]-p[0], pp[1]-p[1])<distance_threshold):
                        exist = True
                        break
                if not exist:
                    extra.append(p)
            for e in extra:
                #if e[1]>250 or (e[0]>400 and e[1]>150):
                 #   exit =exit+1
                if e[1]<200:
                    enter = enter +1
                    count += 1
            #cam_prev_count = in_frame
            #prev_point = _point
        if in_frame < cam_prev_count:
        #determine which points went missing
            #print("missing")
            missing = []
            if in_frame== 0 and prev_point[0][1]<200:
                count -= cam_prev_count
            for p in _point:
                #check if point is any of prev point, if it is, stop checking, move to nets point 
                exist = False    
                for pp in prev_point:
                    if(math.hypot(pp[0]-p[0], pp[1]-p[1])<distance_threshold):
                        exist = True
                        break
                if not exist:
                    missing.append(p)
                    #pqqqqrint("missing2")
            for m in missing:
                #if m[1]>250 or (e[0]>400 and e[1]>150):
                #    exit =exit -1
                #print ("missing")
                if m[1]<200:
                    enter = enter -1
                    count = count -1
            #cam_prev_count = in_frame
            #prev_point = _point
            #count += enter-exit

            if count < 0:
                count = 1
            if prevcount != count:
                update = True
            prevcount = count
        if in_frame ==0:
            enter = 0
            exit = 0

        cam_prev_count = in_frame
        prev_point = _point
        cv2.line (frame, (0, 200), (500, 200), (255, 0, 0), 2)    
        cv2.line (frame, (0, 300), (500, 300), (255, 0, 0), 2)
        cv2.putText(frame, "People: {}".format(count), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "Exit: {}".format(exit), (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "Enter: {}".format(enter), (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.imshow("Security Feed", frame)
    ##    cv2.imshow("Gray", frameDelta)
        
    ##    cv2.imshow("Thresh", thresh)
    ##    cv2.imshow("Frame Delta", frameDelta)
        
        rawCapture.truncate(0)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if (datetime.now().strftime('%H') == '3') and (datetime.now().strftime('%M') == '00'):
                #count will reset every day at 3am
                count = 0
        if key == ord("r"):
                count = 0
        #pushs data into the database at every minute
        if datetime.now().strftime('%S') == '00' and datetime.now().strftime('%M')!=inputtime:
            #pushLogCSV.updateCount(databaseFilepath,count, datetime.now())
            db_connector.pushLog(count, datetime.now())
            inputtime = datetime.now().strftime('%M')
    # cleanup the camera and close any open windows
    ##camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    t = threading.Thread(target=humancounter)
    t.daemon = True
    t.start()
    #print ("stating database")
    #socketio.run(app, debug=False)

    
