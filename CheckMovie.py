import numpy as np
import cv2
import play
def getNum(num):
    
    im = cv2.imread('Number.jpg')
    numLoc =[[460,970],[112,60],[112,280],[112,500],[112,730],[112,970],[460,60],[460,280],[460,500],[460,730]]
    numImg = im[numLoc[num][0]:numLoc[num][0]+300,numLoc[num][1]:numLoc[num][1]+180]
    resImg = np.empty((50,30,3), dtype = int)
    for H in range(50):
        for W in range(30):
            resImg[H][W][0] = numImg[H*6:H*6+6,W*6:W*6+6,0].mean()
            resImg[H][W][1] = numImg[H*6:H*6+6,W*6:W*6+6,1].mean()
            resImg[H][W][2] = numImg[H*6:H*6+6,W*6:W*6+6,2].mean()
    return resImg

def prossesingMovie(image,W,H):
    prossesingImg = image[58:86,82:110]
    num = play.play(prossesingImg)
    image[58,81:111]=[255,255,255]
    image[87,81:111]=[255,255,255]
    image[57:87, 82]=[255,255,255]
    image[57:87,111]=[255,255,255]
    image[10:60,150:180] = getNum(num)


path = 'learnNum.mp4'
cap = cv2.VideoCapture(path)
Width = int(cap.get(3))
Height = int(cap.get(4))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M','4','S','2')  #fourccを定義
out = cv2.VideoWriter('output.wmv',fourcc, 20.0, (Width,Height))  #動画書込準備
count=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        prossesingMovie(frame,Width,Height)
        # write the flipped frame　　
        out.write(frame)           #output.aviにframe毎書込み
        # cv2.imshow('frame',frame)  #反転frameを表示
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    if count == 1000:
        cv2.imwrite('print.jpg', frame)
    
    count +=1
    if count % 20 == 0:
        print(count)
    else:
        print('{},'.format(count),)
    
# Release everything if job is finished
cap.release()
out.release()  #書込み開放
cv2.destroyAllWindows()

print('finish')