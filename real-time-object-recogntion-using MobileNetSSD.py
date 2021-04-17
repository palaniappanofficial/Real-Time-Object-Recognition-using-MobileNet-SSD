import cv2
import numpy as np
import time 
import imutils

prototxt="MobileNetSSD_deploy.prototxt"
model="MobileNetSSD_deploy.caffemodel"
confThresh=0.2

CLASSES=["background","aeroplane","bicycle","bird","boat",
         "bottle","bus","car","cat","chair","cow","diningtable",
         "dog","horse","motorbike","person","pottedplant","ship",
         "sofa","train","tvmonitor"]

COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))

print("Loading Model.....")
net=cv2.dnn.readNetFromCaffe(prototxt,model)
print("Model Successfully Loaded")
print("Starting Camera Feed.....")
vs=cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _,frame=vs.read()
    frame=imutils.resize(frame,width=500)
    (h,w)=frame.shape[:2]
    imResize=cv2.resize(frame,(300,300))
    blob=cv2.dnn.blobFromImage(imResize,0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()
    detshape=detections.shape[2]
    for i in np.arange(0,detshape):
        confidence=detections[0,0,i,2]
        if confidence>confThresh:
            idx=int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startx,starty,endx,endy)=box.astype("int")
            data="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startx,starty),(endx,endy),COLORS[idx],2)
            if starty-15>15:
                y=starty-45
            else:
                starty+15
            cv2.putText(frame,data,(startx,y),cv2.FONT_HERSHEY_SIMPLEX,1,COLORS[idx],2)
    cv2.imshow("Recognised Object",frame)
    key=cv2.waitKey(1)
    if key==27:
        break
vs.release()
cv2.destroyAllWindows()








    

