import glob
import os
import numpy as np
import sys

current_dir = "data/JPEGImages"
split_pct = 10
file_train = open("data/ImageSets/train.txt", "w")  
file_val = open("data/ImageSets/test.txt", "w")  
counter = 1  
index_test = round(100 / split_pct)  
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if counter == index_test:
                counter = 1
                file_val.write(title + "\n")
        else:
                file_train.write(title+ "\n")
                counter = counter + 1
file_train.close()
file_val.close()