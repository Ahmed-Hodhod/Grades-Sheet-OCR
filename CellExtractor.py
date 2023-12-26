
import cv2
import numpy as np
import subprocess
import TableExtractor as te
import os 

class CellExtractor:
    def __init__(self, image, order):
        self.order = order #  the order of the column
        self.image = image


    def execute(self,contours=None):
        if contours is None:
            self.grayscale_image()
            self.threshold_image()
            self.invert_image()
            self.keep_only_horizontal_lines()
            self.find_contours()

        contours=  self.convert_contours_to_bounding_boxes(contours)  
    
        return contours if self.order == 0 else None

    def grayscale_image(self):
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image  = cv2.adaptiveThreshold(cv2.medianBlur(self.gray_image,3),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def erode_vertical_lines(self):
        hor = np.array([[1,1,1,1,1,1]])
      
        image = cv2.erode(self.inverted_image, hor, iterations=20)
        image = cv2.dilate(image, hor, iterations=10)

        # remove the short horizontal lines and keep the longer ones 
        kernel = np.ones((3,15 ), np.uint8)
        self.vertical_lines_eroded_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        
    def keep_only_horizontal_lines(self):
        # handle the horizontal lines with a slope 
        size = (3,3)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)
        processed_image = cv2.dilate(self.inverted_image, kernel, iterations = 1)

        # keep only the horiztonal liness
        hor = np.array([[1,1,1,1,1,1]])
        processed_image = cv2.erode(processed_image, hor, iterations=30)
        processed_image = cv2.dilate(processed_image, hor, iterations=20)

        # merge double lines (very close)
        vert = np.array([[1],[1]])
        processed_image = cv2.dilate(processed_image, vert, iterations=1)
        self.horiozontal_lines = processed_image

    def find_contours(self):

        #self.image= cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        result = cv2.findContours(self.horiozontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #self.contours = sorted(result, key=lambda x: cv2.boundingRect(x)[0])
        self.contours = result[0]
        self.image_with_contours_drawn = self.image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 5)


    def convert_contours_to_bounding_boxes(self,sorted_contours=None):
        if sorted_contours is None:
            self.bounding_boxes = []
            self.image_with_all_bounding_boxes = self.image.copy()
            sorted_contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[1]) 

        self.original_image = self.image
        os.makedirs(f"././processing/column_cells/{self.order}", exist_ok=True)
        for i, contour in enumerate( sorted_contours[0:-1]):
            x, y, w, h = cv2.boundingRect(contour)
            x2, y2, w2, _ = cv2.boundingRect(sorted_contours[i+1])

            if y2 - y < 15:
                continue
            
            #self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (0, y), (self.original_image.shape[1]  , y2), (0, 250, 0),3)
            cropped_image = self.original_image[y:y2 , :  ]
            
            image_slice_path = f"./processing/column_cells/{self.order}/{i}" + ".jpg"
            
            cv2.imwrite(image_slice_path, cropped_image)
        return  sorted_contours
