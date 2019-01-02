
from yolo import Yolo
import cv2




imagepath = "images/dining_table.jpg"

yolofiles = {
  "config":  "yolo-coco/yolov3.cfg",
  "weights": "yolo-coco/yolov3.weights",
  "names":   "yolo-coco/coco.names"
}


yolo = Yolo(yolofiles)

# load our input image and grab its spatial dimensions
image = cv2.imread(imagepath)

yolo.detect(image) #,confidence=0.0,threshold=0.0)
yolo.annotate(image)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
