# -*- coding: cp949 -*-
# -*- coding: utf-8 -*-  # �ѱ� �ּ������� �̰� �ؾ���
import cv2 # opencv ���

image = cv2.imread('solidWhiteCurve.jpg') # �̹��� �б�
cv2.imshow('result',image) # �̹��� ���
cv2.waitKey(0)