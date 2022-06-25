import numpy as np
from PIL import Image
import cv2
import os

IMG_DIR = './imgs'

for img in os.listdir(IMG_DIR):
    #print(img)
    if img == '.DS_Store':
        continue
    img_array = cv2.imread(os.path.join(IMG_DIR,img), cv2.IMREAD_GRAYSCALE)

    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))

    img_array = (img_28x28.flatten())

    img_array  = img_array.reshape(-1,1).T
    
    with open('train.csv', 'ab') as f:
        np.savetxt(f, img_array, delimiter=",")