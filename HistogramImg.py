import cv2
import numpy as np
import matplotlib.pyplot as plt

def Histogram_Image():
    FILENAME1 = '../Images/backgrd1.jpg'
    img1 = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(FILENAME1)

    hist1 = cv2.calcHist( [img1] , [0] , None , [256] , [0,256] )
    hist2,bins = np.histogram( img1.ravel() , 256 , [0,256] )
    hist3 = np.bincount( img1.ravel() , minlength=256 )

    plt.hist( img1.ravel() , 256 , [0,256] ) 
    color = ('b','g','r')
    for i,col in enumerate(color) :
        hist = cv2.calcHist([img2], [i], None, [256], [0,256] )
        plt.plot( hist, color=col)
        plt.xlim( [0,256] )
    plt.show()

def Histogram_Equal():
    FILENAME1 = '../Images/backgrd3.jpg'
    img = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist( img )
    res = np.hstack( (img,equ) )
    cv2.imshow('Equalize',res)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


def Histogram_CLAHE():
    FILENAME1 = '../Images/backgrd3.jpg'
    img = cv2.imread(FILENAME1, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)
    res = np.hstack( (img,img2) )

    cv2.imshow('Equalize',res)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Histogram_CLAHE()    