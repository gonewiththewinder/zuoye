import cv2
import numpy as np


img=cv2.imread("Pictures/520.png",cv2.IMREAD_COLOR)

hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


lower_blue=np.array([100,40,40])

high_blue=np.array([140,255,255])

lower_black=np.array([0,0,0])

high_black=np.array([180,25,30])

mask1=cv2.inRange(hsv_image,lower_blue,high_blue)

mask2=cv2.inRange(hsv_image,lower_black,high_black)

contour1=cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

combine_masks=cv2.bitwise_and(mask1,mask2)

contours,hierachy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

mask11=np.zeros_like(mask2)
cv2.drawContours(mask11,contours,-1,(255,255,255),-1)

kernel=np.ones((6,6),np.uint8)

closed_image=cv2.morphologyEx(mask1,cv2.MORPH_CLOSE,kernel)

contours,hierachy = cv2.findContours(closed_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

mask22=np.zeros_like(closed_image)
cv2.drawContours(mask22,contours,-1,(255,255,255),-1)

combine_masks=cv2.bitwise_and(mask11,mask22)

contours,hierachy = cv2.findContours(combine_masks,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

if len(contours)>0:
    area=cv2.contourArea(contours[0])
    print("封闭的图形面积:",area)   
else :
    print("未找到封闭图形")

cv2.imshow("A",combine_masks)

k=cv2.waitKey(0)

if k==27:
    cv2.destroyAllWindows()