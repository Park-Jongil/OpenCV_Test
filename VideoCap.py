import cv2

def VideoCaptureCode() :
    CheckCam = False
    cap = cv2.VideoCapture(0)
    try:
        if cap.isOpen():
            print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))
            CheckCam = True
    except :
        print( "Except Error")
        
    while CheckCam :
        ret, fram = cap.read()
        if ret:
            gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
            cv2.imshow('video', gray)
            k == cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        else:
            print('error')

    cap.release()
    cv2.destroyAllWindows()	

if __name__ == '__main__':
    VideoCaptureCode()    