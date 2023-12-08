import cv2
import numpy as np
import subprocess
import os 
from PIL import Image
import pytesseract


class OCR:

    def __init__(self, cells_path):
        self.cells_path = cells_path
        self.column = []

    def execute(self):
        self.convert_cell_to_text()
        self.generate_csv_file()

    def convert_cell_to_text(self):
        files = os.listdir(self.cells_path)

        # Loop over each file
        for file in files:
            # Create the full path to the file
            file_path = os.path.join(self.cells_path, file)
            
            # Check if the path is a file (not a subdirectory)
            if os.path.isfile(file_path):
                print(file_path)
                cell_text = self.get_result_from_tersseract(file_path)
                self.column.append(cell_text)
        
        


    def get_result_from_tersseract(self, image_path):
        #output = subprocess.getoutput('tesseract ' + image_path + ' - -l eng --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789().calmg* "')
        
        image = Image.open(image_path)
        output= pytesseract.image_to_string(image)
        #output = output.strip()
        return output
    
    def generate_csv_file(self):
        with open(f"{self.cells_path}/output.csv", "w") as f:
            for row in self.column:
                f.write(",".join(row) + "\n")
        