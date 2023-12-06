import numpy as np
from skimage.color import rgb2gray,rgba2rgb
from skimage.measure import find_contours
from skimage import transform, feature, color , morphology , measure, segmentation, filters, draw, io
from commonfunctions import * 
from skimage.draw import rectangle
from PIL import Image




class TableExtractor:

    def __init__(self, image_path):
        self.image_path = image_path

    def execute(self):
        self.read_image()
        self.store_process_image("0_original.jpg", self.image)
        self.convert_image_to_grayscale()
        self.store_process_image("1_grayscaled.jpg", self.grayscale_image)
        self.threshold_image()
        self.store_process_image("3_thresholded.jpg", self.thresholded_image)
        self.invert_image()
        self.store_process_image("4_inverteded.jpg", self.inverted_image)
        self.dilate_image()
        self.store_process_image("5_dialateded.jpg", self.dilated_image)
        self.find_contours()
        self.store_process_image("6_all_contours.jpg", self.image_with_all_contours)
        # self.filter_contours_and_leave_only_rectangles()
        # self.store_process_image("7_only_rectangular_contours.jpg", self.image_with_only_rectangular_contours)
        # self.find_largest_contour_by_area()
        # self.store_process_image("8_contour_with_max_area.jpg", self.image_with_contour_with_max_area)
        # self.order_points_in_the_contour_with_max_area()
        # self.store_process_image("9_with_4_corner_points_plotted.jpg", self.image_with_points_plotted)
        # self.calculate_new_width_and_height_of_image()
        # self.apply_perspective_transform()
        # self.store_process_image("10_perspective_corrected.jpg", self.perspective_corrected_image)
        # self.add_10_percent_padding()
        # self.store_process_image("11_perspective_corrected_with_padding.jpg", self.perspective_corrected_image_with_padding)
        # return self.perspective_corrected_image_with_padding

    def read_image(self):
        self.image = io.imread(self.image_path)

    def convert_image_to_grayscale(self):
        self.grayscale_image = rgb2gray(self.image)# cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    def blur_image(self):
        self.blurred_image  = filters.gaussian(self.grayscale_image, sigma=5) #cv2.blur(self.grayscale_image, (5, 5))

    def threshold_image(self):
        self.thresholded_image = self.grayscale_image > filters.threshold_otsu(self.grayscale_image) # cv2.threshold(self.grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def invert_image(self):
        self.inverted_image  = np.invert(self.thresholded_image)#cv2.bitwise_not(self.thresholded_image)

    def dilate_image(self):

        # Finally, apply a dilation operation to enlarge foreground regions
        self.dilated_image = morphology.dilation(self.inverted_image, morphology.square(13))

    def find_contours(self):
        # Find contours in the dilated binary image
        self.contours = measure.find_contours(self.dilated_image, level=0.8)

        # Create an empty image to draw contours on
        self.image_with_all_contours = np.zeros_like(self.dilated_image, dtype=np.uint8)

        # Draw each contour on the image
        for contour in self.contours:
            rr, cc = draw.polygon(contour[:, 0], contour[:, 1])
            self.image_with_all_contours[rr, cc] =1



    # def filter_contours_and_leave_only_rectangles(self):
    #     self.rectangular_contours = []
    #     for contour in self.contours:
    #         peri = cv2.arcLength(contour, True)
    #         approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    #         if len(approx) == 4:
    #             self.rectangular_contours.append(approx)
    #     self.image_with_only_rectangular_contours = self.image.copy()
    #     cv2.drawContours(self.image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)

    # def find_largest_contour_by_area(self):
    #     max_area = 0
    #     self.contour_with_max_area = None
    #     for contour in self.rectangular_contours:
    #         area = cv2.contourArea(contour)
    #         if area > max_area:
    #             max_area = area
    #             self.contour_with_max_area = contour
    #     self.image_with_contour_with_max_area = self.image.copy()
    #     cv2.drawContours(self.image_with_contour_with_max_area, [self.contour_with_max_area], -1, (0, 255, 0), 3)

    # def order_points_in_the_contour_with_max_area(self):
    #     self.contour_with_max_area_ordered = self.order_points(self.contour_with_max_area)
    #     self.image_with_points_plotted = self.image.copy()
    #     for point in self.contour_with_max_area_ordered:
    #         point_coordinates = (int(point[0]), int(point[1]))
    #         self.image_with_points_plotted = cv2.circle(self.image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)

    # def calculate_new_width_and_height_of_image(self):
    #     existing_image_width = self.image.shape[1]
    #     existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
        
    #     distance_between_top_left_and_top_right = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[1])
    #     distance_between_top_left_and_bottom_left = self.calculateDistanceBetween2Points(self.contour_with_max_area_ordered[0], self.contour_with_max_area_ordered[3])

    #     aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

    #     self.new_image_width = existing_image_width_reduced_by_10_percent
    #     self.new_image_height = int(self.new_image_width * aspect_ratio)

    # def apply_perspective_transform(self):
    #     pts1 = np.float32(self.contour_with_max_area_ordered)
    #     pts2 = np.float32([[0, 0], [self.new_image_width, 0], [self.new_image_width, self.new_image_height], [0, self.new_image_height]])
    #     matrix = cv2.getPerspectiveTransform(pts1, pts2)
    #     self.perspective_corrected_image = cv2.warpPerspective(self.image, matrix, (self.new_image_width, self.new_image_height))

    # def add_10_percent_padding(self):
    #     image_height = self.image.shape[0]
    #     padding = int(image_height * 0.1)
    #     self.perspective_corrected_image_with_padding = cv2.copyMakeBorder(self.perspective_corrected_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # def draw_contours(self):
    #     self.image_with_contours = self.image.copy()
    #     cv2.drawContours(self.image_with_contours,  [ self.contour_with_max_area ], -1, (0, 255, 0), 1)

    # def calculateDistanceBetween2Points(self, p1, p2):
    #     dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    #     return dis
    
    # def order_points(self, pts):
    #     # initialzie a list of coordinates that will be ordered
    #     # such that the first entry in the list is the top-left,
    #     # the second entry is the top-right, the third is the
    #     # bottom-right, and the fourth is the bottom-left
    #     pts = pts.reshape(4, 2)
    #     rect = np.zeros((4, 2), dtype="float32")

    #     # the top-left point will have the smallest sum, whereas
    #     # the bottom-right point will have the largest sum
    #     s = pts.sum(axis=1)
    #     rect[0] = pts[np.argmin(s)]
    #     rect[2] = pts[np.argmax(s)]

    #     # now, compute the difference between the points, the
    #     # top-right point will have the smallest difference,
    #     # whereas the bottom-left will have the largest difference
    #     diff = np.diff(pts, axis=1)
    #     rect[1] = pts[np.argmin(diff)]
    #     rect[3] = pts[np.argmax(diff)]

    #     # return the ordered coordinates
    #     return rect
    
    def store_process_image(self, file_name, image):
        path = "./process_images/table_extractor/" + file_name
        # Convert the image to uint8 format (if it's not already)
        image_uint8 = (image * 255).astype(np.uint8)

        # Create a Pillow Image object
        image_pillow = Image.fromarray(image_uint8)

        # Save the image using Pillow
        image_pillow.save(path)

        
        