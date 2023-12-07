import OcrToTableTool as ottt
import TableExtractor as te
import ColumnExtractor as ce 
import cv2
import time



for i in range (1,25):
    print(i)
    path_to_image = f"grade_sheet/{i}.jpg"
    table_extractor = te.TableExtractor(path_to_image)
    perspective_corrected_image = table_extractor.execute()
    #cv2.imshow("perspective_corrected_image", perspective_corrected_image)

    # lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
    # image_without_lines = lines_remover.execute()
    # #cv2.imshow("image_without_lines", image_without_lines)

    # ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image, i)
    # image_with_contours = ocr_tool.execute()

    #cv2.imwrite(f"./column_images/{i}.jpg", image_with_contours)

    #time.sleep(3)

    column_extractor = ce.ColumnExtractor(perspective_corrected_image, i)
    image_with_all_bounding_boxes = column_extractor.execute()
    cv2.imwrite(f"./column_images/{i}.jpg", image_with_all_bounding_boxes)


    ##################################### test #######################
    #ocr_tool.extract_first_column()
###################### test #################################


# cv2.waitKey(0)
# cv2.destroyAllWindows()