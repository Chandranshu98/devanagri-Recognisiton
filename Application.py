#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from keras.models import load_model
import numpy as np
from collections import deque


# In[2]:


model =load_model('dev.h5')
print(model)


# In[3]:


letter_count={0:'check',1:'ka',2:'kha',3:'ga',4:'gha',5:'kna',
             6:'cha',7:'chha',8:'ja',9:'jha',10:'yna',
             11:'ta(tamatar) ',12:'ttha (thanda)',13:'da',14:'dha',15:'anda',
             16:'ta (tabala)',17:'tha(thoda)',18:'da',19:'dha',20:'na',
             21:'pa',22:'fa',23:'ba',24:'bha',25:'ma',26:'yaw',27:'ra',28:'la',29:'waw',30:'sha',31:'shha(shatkon)',32:'sa',33:'ha',34:'ksh',35:'tra',36:'gya',37:'check'}


# In[ ]:





# In[4]:


def keras_predict(model1,image):
    processed = keras_process_image(image)
    print("processed: "+ str(processed.shape))
    pred_prob=model.predict(processed)[0]
    pred_class=list(pred_prob).index(max(pred_prob))
    return max(pred_prob),pred_class

def keras_process_image(img):
    x=32
    y=32
    img =cv2.resize(img,(x,y))
    img=np.array(img, dtype=np.float32)
    img=np.reshape(img,(-1,x,y,1))
    return img
    


# In[ ]:


cap = cv2.VideoCapture(0)
Lower_blue= np.array([110,50,50])
Upper_blue= np.array([130,255,255])
pred_class=0
pts=deque(maxlen=512)
blackboard = np.zeros((480,640,3), dtype=np.uint8)
digit = np.zeros((200,200,3), dtype=np.uint8)
while(cap.isOpened()):
    ret, img = cap.read()
    img=cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV, Lower_blue, Upper_blue)
    blur = cv2.medianBlur(mask, 15)
    blur = cv2.GaussianBlur(blur, (5,5),0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts =  cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    center = None
    if len(cnts)>=1:
        contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(contour)>250:
            ((x,y),radius)=cv2.minEnclosingCircle(contour)
            cv2.circle(img, (int(x), int(y)), int(radius), (0,255,255) ,2)
            cv2.circle(img, center, 5, (0,0,255), -1)
            M= cv2.moments(contour)
            center=(int(M['m10']/ M['m00']), int ( M['m01'] / M['m00']))
            pts.appendleft(center)
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard, pts[i-1], pts[i], (255,255,255), 10)
                cv2.line(img, pts[i-1],pts[i], (0,0,255),5)
    elif len(cnts)==0:
        if len(pts)!=[]:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1=cv2.medianBlur(blackboard_gray, 15)
            blur1 = cv2.GaussianBlur(blur1, (5,5),0)
            thresh = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts=cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
            if len(blackboard_cnts) >=1:
                cnt = max(blackboard_cnts, key=cv2.contourArea)
                print(cv2.contourArea(cnt))
                if cv2.contourArea(cnt) > 2000:
                    x ,y, w, h= cv2.boundingRect(cnt)
                    digit= blackboard_gray[y:y +h, x:x +w]
                    pred_prob, pred_class = keras.predict(model1, digit)
                    print(pred_class, pred_prob)
        pts = deque(maxlen =512)
        blackboard= np.zeros((480,640,3), dtype=np.uint8)
    cv2.putText(img, "Conv Network : " + str(letter_count[pred_class]), (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)
    k =cv2.waitKey(10)
    if k == 27:
        break
                    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




