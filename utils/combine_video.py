import numpy as np
import cv2

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter()
#out.open('combined.avi', fourcc, 1, (1280*2, 720*2), True)
out.open('combined.avi', fourcc, 30, (1280, 720), True)

cap1 = cv2.VideoCapture('../md_selfdata_rainy_night.avi')
cap2 = cv2.VideoCapture('../output/rainy_night.avi')

title1 = 'First'
title2 = 'Second'
while(cap1.isOpened() and cap2.isOpened()):

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 == True: 
        border_f1 = cv2.copyMakeBorder(frame1, 80, 30, 30, 30,cv2.BORDER_CONSTANT, value=[255,255,255])
        border_f2 = cv2.copyMakeBorder(frame2, 80, 30, 30, 30,cv2.BORDER_CONSTANT, value=[255,255,255])
        both = np.concatenate((border_f1, border_f2), axis=1)
        cv2.putText(both, title1, (int(0.25*1280*2), 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, 0)
        cv2.putText(both, title2, (int(0.75*1280*2), 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, 0)
        both = cv2.resize(both, (1280,720), interpolation=cv2.INTER_AREA)
        cv2.imshow('Frame', both)
        out.write(both)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            print("video saved")
            break


    else: 
        break

out.release()


cv2.waitKey(0)
cv2.destroyAllWindows()

