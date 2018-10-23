import cv2

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

if __name__ == '__main__':
    showImage()    
