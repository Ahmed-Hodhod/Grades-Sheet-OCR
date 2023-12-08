import OcrToTableTool as ottt
import TableExtractor as te
import ColumnExtractor as ce 
import CellExtractor as cell 
import OCR as ocr 
import Classifier
import os 

import cv2
import time
import numpy as np



for i in range (1,2):
    pass

    # # print(i)
    # path_to_image = f"grade_sheet/{i}.jpg"
    # table_extractor = te.TableExtractor(path_to_image)
    # perspective_corrected_image = table_extractor.execute()
    # #cv2.imshow("perspective_corrected_image", perspective_corrected_image)
  
    # column_extractor = ce.ColumnExtractor(perspective_corrected_image, i)
    # image_with_all_bounding_boxes = column_extractor.execute()
    # cv2.imwrite(f"./column_images/{i}.jpg", image_with_all_bounding_boxes)
  
    image = np.array(cv2.imread(f"./image_columns/{i}_col_3.jpg"))
    cell_extractor = cell.CellExtractor(image)
    image_without_lines = cell_extractor.execute(i)

    # cv2.imwrite(f"./images_without_lines/{i}.jpg", image_without_lines)
    # path = f"./Cells/{i}"
    # text_converter = ocr.OCR(path)
    # text_converter.execute()



def clean_cell(img):
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(cv2.medianBlur(img,1),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    size = (3,3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = cv2.bitwise_not(img)
    return img


classifier = Classifier.Classifier(path_to_dataset="./digits_dataset/")
classifier.train_model() 
tokens = {'a':1, 'b':2,'c':3,'d':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9}

os.makedirs(f"cleaned/", exist_ok=True)
for i in range(1,2):
        files = os.listdir(f"./Cells/{i}")
        files = sorted(files)
        # Loop over each file
        for file in files:
            # Create the full path to the file
            img_path = os.path.join(f"./Cells/{i}", file)
            
            # Check if the path is a file (not a subdirectory)
            if os.path.isfile(img_path):
               img = cv2.imread(img_path)
               img = clean_cell(img)
               cv2.imwrite(f"cleaned/{file}.jpg", img)
               prediction = classifier.classify(img)
               print(file, tokens[prediction[0]])



