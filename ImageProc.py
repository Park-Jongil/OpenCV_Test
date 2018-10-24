import cv2
import numpy as np
import matplotlib.pyplot as plt

def showImage():
    FILENAME1 = '../Images/backgrd.jpg'
    FILENAME2 = '../Images/backgrd1.jpg'
    img1 = cv2.imread(FILENAME1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(FILENAME2, cv2.IMREAD_COLOR)
    addimg1 = img1 + img2
    addimg2 = cv2.add( img1 , img2 )
    cv2.imshow('img1+img2', addimg1)  
    cv2.imshow('cv_img1+img2', addimg2)  
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def Track_OnChange(x) :
    pass

def ImageBlending():
    FILENAME1 = '../Images/backgrd.jpg'
    FILENAME2 = '../Images/backgrd1.jpg'
    img1 = cv2.imread(FILENAME1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(FILENAME2, cv2.IMREAD_COLOR)

    cv2.namedWindow( "ImgPanel" )
    cv2.createTrackbar('Mixing',"ImgPanel",0,100,Track_OnChange)
    mix = cv2.getTrackbarPos('Mixing',"ImgPanel")
    while True :
        img = cv2.addWeighted( img1, float(100-mix)/100, img2, float(mix)/100,0 )
        cv2.imshow('ImgPanel',img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:         # wait for ESC key to exit
            break
        mix = cv2.getTrackbarPos('Mixing',"ImgPanel")
    cv2.destroyAllWindows()

def ImageFiltering():
    FILENAME1 = '../Images/backgrd1.jpg'
    img = cv2.imread(FILENAME1, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array( [110,100,100] )
    upper_blue = np.array( [130,255,255] )
    lower_green = np.array( [ 50,100,100] )
    upper_green = np.array( [ 70,255,255] )
    lower_red = np.array( [-10,100,100] )
    upper_red = np.array( [ 10,255,255] )

    mask_blue  = cv2.inRange( hsv, lower_blue, upper_blue )
    mask_green = cv2.inRange( hsv, lower_green, upper_green )
    mask_red   = cv2.inRange( hsv, lower_red, upper_red )

    res1 = cv2.bitwise_and(img,img,mask=mask_blue)
    res2 = cv2.bitwise_and(img,img,mask=mask_green)
    res3 = cv2.bitwise_and(img,img,mask=mask_red)

    cv2.imshow( 'Original' , img )
    cv2.imshow( 'BLUE' , res1 )
    cv2.imshow( 'GREEN' , res2 )
    cv2.imshow( 'RED' , res3 )
    
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()



if __name__ == '__main__':
    ImageFiltering()    
