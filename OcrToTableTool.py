import cv2
import numpy as np
import subprocess

class OcrToTableTool:
    def __init__(self, original_image):
        self.original_image = original_image
        self.grey = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        self.thresholded_image = cv2.threshold(self.grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def execute(self):
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()
        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()

    def threshold_image(self):
        return cv2.threshold(self.grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    def convert_image_to_grayscale(self):
        return cv2.cvtColor(self.image, self.dilated_image)
    def dilate_image(self):
        kernel_to_remove_gaps_between_words = np.array([
                [1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1]
        ])
        self.dilated_image = cv2.dilate(self.thresholded_image, kernel_to_remove_gaps_between_words, iterations=5)
        simple_kernel = np.ones((5,5), np.uint8)
        self.dilated_image = cv2.dilate(self.dilated_image, simple_kernel, iterations=2)
    
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
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 5)
   
    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)
    
    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])
    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [ bounding_box ]
        self.rows.append(current_row)
    def sort_all_rows_by_x_coordinate(self):
        self.first_col =[]
        self.third_col = []
        for row in self.rows:
            row.sort(key=lambda x: x[0])
            self.first_col.append(row[0])
            self.third_col.append(row[2]) if len(row) > 2 else None
            self.first_col.append(list(row[0]))
            self.third_col.append(list(row[2])) if len(row) > 2 else None

    # def extract_first_column(self):
    #     print(self.first_col)


    #     data = np.array(self.first_col)
    #     # Calculate the first and third quartiles (Q1 and Q3)
    #     hist, edges = np.histogram(data[:,0], bins=30)

    #     # Identify the bin with the maximum frequency
    #     max_freq_bin_index = np.argmax(hist)
    #     most_frequent_range = (edges[max_freq_bin_index], edges[max_freq_bin_index + 1])
    #     print(most_frequent_range)

    #     x0 = data[:,0]
    #     condition1 = x0 >= edges[max_freq_bin_index] - 20 
    #     condition2 = x0 <= edges[max_freq_bin_index] + 20
    #     filter = condition1 & condition2
    #     removed_data = data[~filter]
    #     filtered_data= data[filter]
    #     print("x0: ", x0)
    #     print("data", data , "\n", len(data))
    #     print("filtered", filtered_data, "\n", len(filtered_data))
    #     print("removed", removed_data, "\n" , len(removed_data))
    #     for i,box in enumerate(filtered_data):
    #         x, y, w, h = box
    #         y = y - 5
    #         cropped_image = self.original_image[y:y+h, x:x+w]
    #         image_slice_path = "./temp_test/img_" + str(i) + ".jpg"
    #         cv2.imwrite(image_slice_path, cropped_image)


    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        current_row = []
        image_number = 0
        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                y = y - 5
                cropped_image = self.original_image[y:y+h, x:x+w]
                image_slice_path = "./ocr_slices/img_" + str(image_number) + ".jpg"
                cv2.imwrite(image_slice_path, cropped_image)
                results_from_ocr = self.get_result_from_tersseract(image_slice_path)
                current_row.append(results_from_ocr)
                image_number += 1
            self.table.append(current_row)
            current_row = []
    def get_result_from_tersseract(self, image_path):
        output = subprocess.getoutput('tesseract ' + image_path + ' - -l eng --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "')
        output = output.strip()
        return output
    def generate_csv_file(self):
        with open("output.csv", "w") as f:
            for row in self.table:
                f.write(",".join(row) + "\n")
    def store_process_image(self, file_name, image):
        path = "./process_images/ocr_table_tool/" + file_name
        cv2.imwrite(path, image)