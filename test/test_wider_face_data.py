import cv2
from bestvision.datasets.wider_face import WiderFace
from bestvision.utils.draw import draw_bboxes, draw_keypoints

if __name__ == "__main__":
    
    data = WiderFace(data_dir="F:\\sujz\\Data\\WIDER_FACE\\WIDER_train")

    for item in data:
        image = draw_bboxes(item['image'], item['bboxes'])
        draw_keypoints(image, item['keypoints'])
        cv2.imshow("v", image)
        cv2.waitKey()
         