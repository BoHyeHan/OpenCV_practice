# -*- coding: utf-8 -*- # 한글 주석쓰려면 이거 해야함
import cv2  # opencv 사용
import numpy as np


def grayscale(img):  # 흑백이미지로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):  # Canny 알고리즘
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 셋팅

    mask = np.zeros_like(img)  # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):  # 선 그리기
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):  # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    return line_img

'''img: 8bit 즉 1채널인 흑백이미지만 가능 // 보통 Canny로 edge를 찾은 후에 이 함수를 적용하므로 이미 흑백인 상태
rho: hough에서 p값(원점에서 직선까지의 거리)을 얼마만큼 증가시킬 것인지 // 보통 1
theta: 보통 각도 값을 입력한 후 pi/180 을 곱한다. 각도 값은 [0:180]의 값을 입력한다.
        theta 역시 얼마만큼 증가시키면서 조사할 것인지를 묻는 것. 보통 1
threshold: 
np.array([]): 그냥 빈 array
min_line_length: 최소 길이의 선, 단위는 픽셀
max_line_gap: 선 위의 점들 사이 최대 거리. 즉 점 사이의 거리가 이 값보다 크면 나와는 다른 선으로 간주하겠다는 의미
            (잘 이해 안감)
output: 선분들의 시작점과 끝점에 대한 좌표 값'''


def weighted_img(img, initial_img, α=1, β=1., λ=0.):  # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, α, img, β, λ)

'''addWeighted(img1, alpha, img2, beta, gamma)
alpha: img1의 가중치 // img2: img1과 같은 size, 같은 channel의 이미지 // beta: img2의 가중치'''


image = cv2.imread('solidWhiteCurve.jpg')  # 이미지 읽기
height, width = image.shape[:2]  # 이미지 높이, 너비

gray_img = grayscale(image)  # 흑백이미지로 변환

blur_img = gaussian_blur(gray_img, 3)  # Blur 효과

canny_img = canny(blur_img, 70, 210)  # Canny edge 알고리즘

vertices = np.array(
    [[(50, height), (width / 2 - 45, height / 2 + 60), (width / 2 + 45, height / 2 + 60), (width - 50, height)]],
    dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices)  # ROI 설정

hough_img = hough_lines(ROI_img, 1, 1 * np.pi / 180, 30, 10, 20)  # 허프 변환

result = weighted_img(hough_img, image)  # 원본 이미지에 검출된 선 overlap
cv2.imshow('result', result)  # 결과 이미지 출력
cv2.waitKey(0)

