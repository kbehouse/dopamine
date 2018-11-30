import cv2,os
import glob
import numpy as np
images = glob.glob( './' + "*.jpg"  )


def cut_2_same_wh(img):
    # print('img.shape=', img.shape) 
    shape = np.shape(img)
    print('shape =', shape)
    print('img.shape[0] = ', shape[0], ', img.shape[1] = ', shape[1] )
    if shape[0] > shape[1]:
        return img[:shape[1],:,: ]
    elif shape[0] < shape[1]:
        return img[:,:shape[0],: ]
    else:
        return img
IMG_W_H = 256

for img_path in images:
    # Load an color image in grayscale
    img = cv2.imread(img_path)
    img = cut_2_same_wh(img)
    img = cv2.resize(img, (IMG_W_H, IMG_W_H))
    filename, file_extension = os.path.splitext(img_path)
    
    new_file_name = filename +'_cut_' + str(IMG_W_H)+file_extension
    cv2.imwrite(new_file_name, img)