from importlib.resources import path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re

# paths
# ask user what he/she wants to create: me or not-me images
def get_input():
	while(True):
		user_input = input('Do you want to create images of yourself or not of yourself (e.g. empty room)?\nType me/not: ')
		if user_input == 'me':
			print('You can now take pictures of yourself!')
			return 'me'
		if user_input == 'not':
			print('You can now take pictures of not yourself!')
			return 'not_me'
		else:
			print("Wrong input! Valid input is 'yes' or 'not'.\r")

folder_type = get_input()
path_save = f'/home/gtex/Desktop/cv/img/{folder_type}/'

# define a video capture object
vid = cv2.VideoCapture(0)

# get n: number of pictures
file_names = os.listdir(path_save)
n = 0
for name in file_names:
    n_poss = int(re.search(r'\d+', name).group())
    if n_poss >= n:
        n = n_poss + 1
print('File number n =', n)

while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    
    # save frame
    img_name = f'img{n}.jpg'
    cv2.imwrite(path_save+img_name, frame)
    print(f'Image "{img_name}" saved to {path_save}.')

    # Display the resulting frame
    cv2.namedWindow('frame', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('frame', frame)
    n +=1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
