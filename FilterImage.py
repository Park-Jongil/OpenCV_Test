import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage():
    FILENAME1 = '../Images/backgrd1.jpg'
    img1 = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)

    laplacian = cv2.Laplacian( img1, cv2.CV_64F)
    sobelx = cv2.Sobel(img1, cv2.CV_64F, 1,0, ksize=3)
    sobely = cv2.Sobel(img1, cv2.CV_64F, 0,1, ksize=3)

    plt.subplot(2,2,1), plt.imshow(img1, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2), plt.imshow(laplacian, cmap='gray')
    plt.title('laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3), plt.imshow(sobelx, cmap='gray')
    plt.title('sobelx'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4), plt.imshow(sobely, cmap='gray')
    plt.title('sobely'), plt.xticks([]), plt.yticks([])

    plt.show()

def CannyFilter():
    FILENAME1 = '../Images/backgrd.jpg'
    img1 = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)

    edge1 = cv2.Canny( img1,  50 , 200)
    edge2 = cv2.Canny( img1, 100 , 200)
    edge3 = cv2.Canny( img1, 170 , 200)

    plt.subplot(2,2,1), plt.imshow(img1, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2), plt.imshow(edge1, cmap='gray')
    plt.title('edge1'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3), plt.imshow(edge2, cmap='gray')
    plt.title('edge2'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4), plt.imshow(edge3, cmap='gray')
    plt.title('edge3'), plt.xticks([]), plt.yticks([])

    plt.show()

def showPyramids_UpDown():
    FILENAME1 = '../Images/backgrd1.jpg'
    img1 = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)
    img2 = img1.copy()

    win_titles = [ 'Org' , 'Lv1' , 'Lv2' , 'Lv3']
    g_down = []
    g_down.append(img2)
    g_up = []

    for i in range(3) :
        tmp1 = cv2.pyrDown( img2 )
        g_down.append(tmp1)
        img2 = tmp1

    cv2.imshow( "Lv3" , img2 )
    for i in range(3) :
        img2 = g_down[i+1]
        tmp1 = cv2.pyrUp( img2 )
        g_up.append(tmp1)
        
    for i in range(3) :
        cv2.imshow( win_titles[i] , g_up[i] )

    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

def showPyramids():
    FILENAME1 = '../Images/backgrd.jpg'
    img = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)
    tmp = img.copy()

    win_titles = [ 'Org' , 'Lv1' , 'Lv2' , 'Lv3']
    g_down = []
    g_up = []
    img_shape = []

    g_down.append(tmp)
    img_shape.append(tmp)

    for i in range(3) :
        tmp1 = cv2.pyrDown( tmp )
        g_down.append(tmp1)
        img_shape.append(tmp1.shape)
        tmp = tmp1

    for i in range(3) :
        tmp = g_down[i+1]
        tmp1 = cv2.pyrUp( tmp )
#        tmp = cv2.resize( tmp1, dsize=(img_shape[i][1], img_shape[i][0]),interpolation=cv2.INTER_CUBIC)
        g_up.append(tmp1)
        
    for i in range(3) :
        tmp = cv2.subtract( g_down[i] , g_up[i] )
        cv2.imshow( win_titles[i] , tmp )

    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

def show_Contour():     # 기존이미지에 윤곽선에 색상으로 마킹하는것.
    FILENAME1 = '../Images/backgrd.jpg'
    img = cv2.imread(FILENAME1)
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imggray, 127, 255, 0)
    _, contours , _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours , -1 , (0,0,255), 1)
    cv2.imshow('Thresh',thr)
    cv2.imshow('Contour',img)

    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

def size_Contour():     # 기존이미지에 윤곽선에 색상으로 마킹하는것.
    FILENAME1 = '../Images/backgrd.jpg'
    img = cv2.imread(FILENAME1)
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imggray, 127, 255, 0)
    _, contours , _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour = contours[0]
    mmt = cv2.moments( contour )
    for key, val in mmt.items() :
        print('%s:\t%.5f'%(key,val))
    cx = int(mmt['m10']/mmt['m00'])
    cy = int(mmt['m01']/mmt['m00'])
    print( cx , cy )

def draw_Contour():     # 기존이미지에 윤곽선에 색상으로 마킹하는것.
    FILENAME1 = '../Images/backgrd.jpg'
    img = cv2.imread(FILENAME1)
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, thr = cv2.threshold(imggray, 127, 255, 0)
    _, contours , _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cnt = contours[1]
    area = cv2.contourArea( cnt )
    perimeter = cv2.arcLength( cnt , True )
    cv2.drawContours( img , [cnt] , 0 , (255,0,0) , 1)
    print('Contour 면적 : ',area)
    print('Contour 길이 : ',perimeter)
    cv2.imshow('Contour',img)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


if __name__ == '__main__':
    draw_Contour()    