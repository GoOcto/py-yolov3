
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os


class Yolo:

  def __init__(self,yolofiles):

    # load the COCO class labels our YOLO model was trained on
    self.LABELS = open(yolofiles["names"]).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
      dtype="uint8")

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    self.net = cv2.dnn.readNetFromDarknet(yolofiles["config"], yolofiles["weights"])

    # determine only the *output* layer names that we need from YOLO
    ln = self.net.getLayerNames()
    self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


  def detect(self,image,confidence=0.25,threshold=0.25,gridsize=416):

    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (gridsize, gridsize),
      swapRB=True, crop=False)
    self.net.setInput(blob)
    layerOutputs = self.net.forward(self.ln)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    self.boxes = []
    self.confidences = []
    self.classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
      # loop over each of the detections
      for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confscore = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confscore > confidence:
          # scale the bounding box coordinates back relative to the
          # size of the image, keeping in mind that YOLO actually
          # returns the center (x, y)-coordinates of the bounding
          # box followed by the boxes' width and height
          box = detection[0:4] * np.array([W, H, W, H])
          (centerX, centerY, width, height) = box.astype("int")

          # use the center (x, y)-coordinates to derive the top and
          # and left corner of the bounding box
          x = int(centerX - (width / 2))
          y = int(centerY - (height / 2))

          # update our list of bounding box coordinates, confidences,
          # and class IDs
          self.boxes.append([x, y, int(width), int(height)])
          self.confidences.append(float(confscore))
          self.classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, confidence, threshold )

  def annotate(self,image):

    # ensure at least one detection exists
    if len(self.idxs) > 0:
      # loop over the indexes we are keeping
      for i in self.idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (self.boxes[i][0], self.boxes[i][1])
        (w, h) = (self.boxes[i][2], self.boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in self.COLORS[self.classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(self.LABELS[self.classIDs[i]], self.confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
          0.5, color, 2)
