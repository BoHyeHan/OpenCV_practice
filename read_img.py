# -*- coding: cp949 -*-
# -*- coding: utf-8 -*-  # 한글 주석쓰려면 이거 해야함
import cv2 # opencv 사용

image = cv2.imread('solidWhiteCurve.jpg') # 이미지 읽기
cv2.imshow('result',image) # 이미지 출력
cv2.waitKey(0)