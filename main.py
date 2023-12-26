import TableExtractor as te
import ColumnExtractor as ce 
import CellExtractor as cell 
import os 
import re

import cv2
import numpy as np


# pass the input with an extension
def chop_image_into_cells(path_to_image):
    table_extractor = te.TableExtractor(path_to_image)
    table = table_extractor.execute()
    column_extractor = ce.ColumnExtractor(table)
    image_with_all_bounding_boxes = column_extractor.execute()
    cv2.imwrite(f"./processing/image_with_all_contours_around_columns.jpg", image_with_all_bounding_boxes)

    folder_path = "./processing/image_columns"
    files = os.listdir(folder_path)
    # Sort the list of files alphabetically
    files.sort()

    # Loop over each file in the folder
    number_of_columns=len(files)
    cells =[[]for _ in range(number_of_columns)]

    #contours
    contours=None
    for i,file_name in enumerate(files):
        print(i,file_name)
        # Get the full path of the file
        file_path = os.path.join(folder_path, file_name)

        # Check if the path points to a file (not a subfolder)
        if os.path.isfile(file_path):
            image = np.array(cv2.imread(file_path))
            cell_extractor = cell.CellExtractor(image,i)
            
            contours = cell_extractor.execute(contours)
            #cv2.imwrite(f"./processing/column_{i}_cells_contoured.jpg", column_cells_contoured)


    folder_path = "./processing/column_cells"
    cols = os.listdir(folder_path)
    cols.sort()

    for i,col in enumerate(cols):
        col_path = os.path.join(folder_path, col)
        images = os.listdir(col_path)

        images.sort()


    ####
        images_with_numbers = [(image, int(re.search(r'\d+', image).group())) for image in images]
        # Sort the list based on the extracted integers
        images_with_numbers.sort(key=lambda x: x[1])


    ###

        for j,image in enumerate(images_with_numbers):
            image_path = os.path.join(col_path,image[0])
            if os.path.isfile(image_path):
                img = np.array(cv2.imread(image_path))
                #print(image)
                cells[i].append(img[0])
    return cells



cells  =chop_image_into_cells("grade_sheet/2.jpg")
for i,cell in enumerate(cells):
    print(len(cells[i]))
print(len(cells[1]))