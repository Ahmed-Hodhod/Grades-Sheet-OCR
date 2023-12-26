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


    
    # # print(i)
path_to_image = "grade_sheet/1.jpg"
table_extractor = te.TableExtractor(path_to_image)
table = table_extractor.execute()
column_extractor = ce.ColumnExtractor(table)
image_with_all_bounding_boxes = column_extractor.execute()
cv2.imwrite(f"./images_with_bounding_boxes/1.jpg", image_with_all_bounding_boxes)
  
    # image = np.array(cv2.imread(f"./image_columns/{i}_col_3.jpg"))
    # cell_extractor = cell.CellExtractor(image)
    # image_without_lines = cell_extractor.execute(i)

    # cv2.imwrite(f"./images_without_lines/{i}.jpg", image_without_lines)
    # path = f"./Cells/{i}"
    # text_converter = ocr.OCR(path)
    # text_converter.execute()



