import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage():
    FILENAME = 'Images/Image001.png'
    # 이미지 파일을 읽기 위한 객체를 리턴  인자(이미지 파일 경로, 읽기 방식)
    # cv2.IMREAD_COLOR : 투명한 부분 무시되는 컬러
    # cv2.IMREAD_GRAYSCALE : 흑백 이미지로 로드
    # cv2.IMREAD_UNCHANGED : 알파 채컬을 포함한 이미지 그대로 로드
    image = cv2.imread(FILENAME, cv2.IMREAD_COLOR)
    cv2.namedWindow('model', cv2.WINDOW_NORMAL  )  #윈도우 창의 성격 지정 인자 : (윈도우타이틀, 윈도우 사이즈 플래그) , 사용자가 크기 조절할 수 있는 윈도우 생성
    cv2.imshow('model', image)  # 화면에 이미지 띄우기 인자;(윈도우타이틀, 이미지객체)

    flag = False
    while True :
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            break
        elif k == ord('s'): # wait for 's' key to save
            flag = not flag
            img_show = None
        if flag :
            img_show = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else :
            img_show = image
        cv2.imshow('model', img_show)    
    
    cv2.destroyAllWindows()  # 생성한 윈도우 제거

def showImage_01():
    FILENAME = 'Images/Image001.png'
    image = cv2.imread(FILENAME, cv2.IMREAD_GRAYSCALE)
    plt.imshow( image , cmap='gray' , interpolation='bicubic')
    plt.xticks( [] )
    plt.yticks( [] )
    plt.title('balls')
    plt.show()

def Drawing() :
    img = np.zeros((512,512,3),np.uint8)
    cv2.line(img,(0,0),(511,511),(255,0,0),2)
    cv2.circle(img,(447,63),63,(0,0,255),-1)

    cv2.imshow('Drawing',img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


def Track_OnChange(x) :
    pass

def Trackbar_Test() :
    img = np.zeros((200,512,3),np.uint8)
    cv2.namedWindow('Color_Pallete')
    cv2.createTrackbar('B','Color_Pallete',0,255,Track_OnChange)
    cv2.createTrackbar('G','Color_Pallete',0,255,Track_OnChange)
    cv2.createTrackbar('R','Color_Pallete',0,255,Track_OnChange)
    cv2.createTrackbar('0:OFF\n1:On','Color_Pallete',0,1,Track_OnChange)

    while True :
        cv2.imshow('Color_Pallete',img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            break
        b = cv2.getTrackbarPos('B','Color_Pallete')
        g = cv2.getTrackbarPos('G','Color_Pallete')
        r = cv2.getTrackbarPos('R','Color_Pallete')
        s = cv2.getTrackbarPos('0:OFF\n1:On','Color_Pallete')
        if s==0 :
            img[:] = 0
        else :
            img[:] = [b,g,r]
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Trackbar_Test()    
