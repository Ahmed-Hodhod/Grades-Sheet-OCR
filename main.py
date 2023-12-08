import OcrToTableTool as ottt
import TableExtractor as te
import ColumnExtractor as ce 
import CellExtractor as cell 
import cv2
import time
import numpy as np



for i in range (1,25):
    # # print(i)
    # path_to_image = f"grade_sheet/{i}.jpg"
    # table_extractor = te.TableExtractor(path_to_image)
    # perspective_corrected_image = table_extractor.execute()
    # #cv2.imshow("perspective_corrected_image", perspective_corrected_image)
  
    # column_extractor = ce.ColumnExtractor(perspective_corrected_image, i)
    # image_with_all_bounding_boxes = column_extractor.execute()
    # cv2.imwrite(f"./column_images/{i}.jpg", image_with_all_bounding_boxes)
  
    image = np.array(cv2.imread(f"./image_columns/{i}_col_0.jpg"))
    cell_extractor = cell.CellExtractor(image)
    image_without_lines = cell_extractor.execute(i)

    cv2.imwrite(f"./images_without_lines/{i}.jpg", image_without_lines)

    ##################################### test #######################
    #ocr_tool.extract_first_column()
###################### test #################################


# cv2.waitKey(0)
# cv2.destroyAllWindows()