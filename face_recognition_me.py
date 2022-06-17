import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# local imports
import getmodel

# variables model
shape_ord = (480, 640, 3)

# variables: for the two rectangles with text
x,y,w,h = 0,0,380,80
x2,y2,w2,h2 = 560,400,80,40
txt = 'NO PREDICTION YET'
color = (255,255,255)


# variables for loop/prediction
k = 1
per_pred = 0.00
n_pred = 0
n_true = 0

## create model and load weights
model = getmodel.build_model(shape_ord)
model.load_weights('model_weights/model_weights_dropout3.h5')


# define a video capture object
vid = cv2.VideoCapture(0)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # make prediction
    prediction = model.predict(frame.reshape(1,480,640,3), verbose=0)

    # safe sum of prediction to calculate an average over 20 frames
    n_pred += prediction.sum() #just a single number but need int

    # counter for predicting me in 20 frames
    if prediction.sum() >= 0.9:
        n_true += 1

    # evaluation all 20 frames
    if k == 20:
        per_pred = n_pred/k

        # if i got predicted at least 5 times of 20 frames with more than 90%
        if n_true >= 5:
            txt = f'Welcome Samuel!'
            color = (0,255,0)
            print(f'Welcome Samuel! ({per_pred*100:.2f}%)')
        else:
            txt = f'You seem afk!...'
            color = (0,0,255)
            print(f'You seem to be afk! ({per_pred*100:.2f}%)')

        k=0
        n_pred = 0
        n_true = 0

    
    # Create background rectangle with color and add text
    # top, left
    cv2.rectangle(frame, (x,x), (x + w, y + h), (0,0,0), -1)
    cv2.putText(img=frame, text=txt,org=(x + int(w/10),y + int(h/1.5)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=color, thickness=2)
    
    # bottom, right
    cv2.rectangle(frame, (x2,x2), (x2 + w2, y2 + h2+10), (0,0,0), -1)
    cv2.putText(img=frame, text=f'{per_pred*100:.2f}%',org=(x2 + int(w2/10),y2 + int(h2)+ 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0,255,0), thickness=1)


    # show frame
    cv2.imshow('frame', frame)
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)

    k += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
