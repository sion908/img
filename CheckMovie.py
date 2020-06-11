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

def prossesingMovie(image,W,H,count):
    #中央480x480の画像を使う
    imgVert = [H // 2 - 240 , W // 2 - 240 ]
    prossesingImg = image[imgVert[0] : imgVert[0] + 480 , imgVert[1] : imgVert[1]+480 ]
    num = play.play(prossesingImg,count)
    image[imgVert[0]-20:imgVert[0],:]=[255,255,255]
    image[imgVert[0]+480-20:480+imgVert[0],:]=[255,255,255]
    image[:,imgVert[1]-20:imgVert[1]]=[255,255,255]
    image[:,imgVert[1]-20+480:480+imgVert[1]]=[255,255,255]
    image[10:60,150:180] = getNum(num)


path = 'fs.wmv'
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
        prossesingMovie(frame,Width,Height,count)
        # write the flipped frame　　
        out.write(frame)           #output.aviにframe毎書込み
        # cv2.imshow('frame',frame)  #反転frameを表示
        
    else:
        break
    if count == 100:
        cv2.imwrite('print.jpg', frame)
    # #とりあえず切る
    # break   
    
    count +=1
    if count % 10 == 0:
        print(count)
    else:
        print('{},'.format(count),end='')
    
    
# Release everything if job is finished
cap.release()
out.release()  #書込み開放
cv2.destroyAllWindows()

print('finish')