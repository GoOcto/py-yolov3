


from yolo import Yolo
import cv2
import time


src = "videos/sample.mp4"

yolofiles = {
  "config":  "yolo-coco/yolov3.cfg",
  "weights": "yolo-coco/yolov3.weights",
  "names":   "yolo-coco/coco.names"
}


yolo = Yolo(yolofiles)

src_scale = 1
src_mirror = False
src_skip = 1

vid = cv2.VideoCapture(src)

t0 = time.time()

while True:
  count = src_skip

  while count>=1:
    count -= 1
    grabbed, image = vid.read()
    if not grabbed: break
    if src_mirror: 
      image = cv2.flip(image, 1)
    if src_scale!=1:
      image = cv2.resize(image,(0,0),fx=src_scale,fy=src_scale)

  # Apply yolo
  yolo.detect(image,0.25,0.3)
  yolo.annotate(image)

  list = []
  if len(yolo.idxs) > 0:
    for i in yolo.idxs.flatten():
      list.append( yolo.LABELS[yolo.classIDs[i]] )
      print ( list )

  t1 = time.time()
  elap = t1-t0
  t0 = t1

  fps = round(1/elap)
  text = ("%d fps"%fps)
  cv2.putText(image,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,128,255], 2)

  cv2.imshow('annotated output', image)
  if cv2.waitKey(1) == 27: 
    break  # esc to quit

cv2.destroyAllWindows()
