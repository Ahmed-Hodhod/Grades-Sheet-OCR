import cv2
import numpy as np

class ColumnExtractor:

    def __init__(self, image, order_of_image):
        self.original_image = image
        self.order_of_image = order_of_image
        

    def execute(self):
        self.grayscale_image()
        self.store_process_image("0_grayscaled.jpg", self.grey)
        self.threshold_image()
        self.store_process_image("1_thresholded.jpg", self.column_borders)
        self.invert_image()
        self.store_process_image("2_inverted.jpg", self.inverted_image)
        self.erode_vertical_lines()
        self.store_process_image("3_erode_vertical_lines.jpg", self.vertical_lines_eroded_image)
        self.erode_horizontal_lines()
        self.store_process_image("4_erode_horizontal_lines.jpg", self.column_borders)
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        return self.image_with_all_bounding_boxes


    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.column_borders = cv2.threshold(self.grey, 127, 255, cv2.THRESH_BINARY)[1]

        # test 
        #self.column_borders = cv2.equalizeHist(self.column_borders)

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.column_borders)

    def erode_vertical_lines(self):
        hor = np.array([[1 for _ in range(5)], ])
        image = cv2.erode(self.inverted_image, hor, iterations=10)

        hor = np.array([[1 for _ in range(10)],[1 for _ in range(10)] ])
        self.vertical_lines_eroded_image = cv2.dilate(image, hor, iterations=10)


        # # remove the short horizontal lines and keep the longer ones 
        # kernel = np.ones((1,15 ), np.uint8)
        # self.vertical_lines_eroded_image = cv2.morphologyEx(self.vertical_lines_eroded_image, cv2.MORPH_OPEN, kernel)

    def erode_horizontal_lines(self):
        ver = np.array([[1],
               [1],
               [1],
               [1],
               [1],
               [1],
               [1],
                [1],
               [1],
               [1],
               [1]])
        self.column_borders = cv2.erode(self.inverted_image, ver, iterations=15)
        self.column_borders = cv2.dilate(self.column_borders, ver, iterations=50)
        

    def dilate_image(self):
        
        kernel_to_remove_gaps_between_words = np.array([
                [1] for _ in range(4)   
        ])
        self.dilated_image = cv2.dilate(self.column_borders, kernel_to_remove_gaps_between_words, iterations=5)
        # simple_kernel = np.ones((5,5), np.uint8)
        # self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=2)


    
    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        #self.contours = sorted(result, key=lambda x: cv2.boundingRect(x)[0])
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
    
    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.original_image_with_contours = self.original_image.copy()
        cv2.drawContours(self.original_image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        sorted_contours = sorted(self.contours, key=lambda c: cv2.boundingRect(c)[0]) 

        self.original_image = self.original_image
 
        print(len(sorted_contours))
        for i, contour in enumerate( sorted_contours[0:-1]):
            
            x, y, w, h = cv2.boundingRect(contour)
            x2, y2, w2, _ = cv2.boundingRect(sorted_contours[i+1])

            if x2 - x < 15:
                continue 

            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x + 20, 0), (x2 -20  , self.original_image.shape[0]), (0, 250, 0),5)
            
            cropped_image = self.original_image[: , x +3:x2 + w2//2   ]
            image_slice_path = f"./image_columns/{self.order_of_image}_col_" + str(i) + ".jpg"
            cv2.imwrite(image_slice_path, cropped_image)

        # #cropped_image = self.vertical_lines_eroded_image[: , x  :x2 + w2 ]
        # cropped_image = self.vertical_lines_eroded_image
        # image_slice_path = f"./image_rows/{self.order_of_image}_rows_" + str(i) + ".jpg"
        # cv2.imwrite(image_slice_path, cropped_image)


    def store_process_image(self, file_name, image):
        path = "./ColumnExtractor/" + file_name
        cv2.imwrite(path, image)