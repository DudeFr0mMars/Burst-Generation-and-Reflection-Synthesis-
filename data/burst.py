import numpy as np
import cv2
import random
import os

def create(root):
    for root, _, fnames in sorted(os.walk(root)):
        for fname in fnames:
            path=root+'/'+fname
            img = cv2.imread(path)
            zoom_factor=1
            rows, cols, ch = img.shape
            pts1 = np.float32([[cols*.25, rows*.95],
                       [cols*.90, rows*.95],
                       [cols*.10, 0],
                       [cols, 0]])
            pts2 = np.float32([[cols*0.1, rows],
                               [cols,     rows],
                               [0,        0],
                               [cols,     0]])

            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(img, M, (cols, rows))
            angle_list=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
            for i in range(len(angle_list)):
                angle_size=random.choice(angle_list)
                height, width = img.shape[:2]  # It's also the final desired shape
                new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
                y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
                y2, x2 = y1 + height, x1 + width
                bbox = np.array([y1, x1, y2, x2])
                bbox = (bbox / zoom_factor).astype(np.int)
                y1, x1, y2, x2 = bbox
                cropped_img = img[y1:y2, x1:x2]
                resize_height, resize_width = min(new_height, height), min(new_width, width)
                pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
                pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
                pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)
                result = cv2.resize(cropped_img, (resize_width, resize_height))
                result = np.pad(result, pad_spec, mode='constant')
                assert result.shape[0] == height and result.shape[1] == width
                M = cv2.getPerspectiveTransform(pts1, pts2)
                dst = cv2.warpPerspective(result, M, (cols, rows))
                row, col,chan = result.shape
                center = tuple(np.array([row, col]) / 2)
                M = np.float32([[1, 0, 100], [0, 1, 50]])
                rot_mat = cv2.getRotationMatrix2D(center, angle_size, 1.0)
                new_image = cv2.warpAffine(result,rot_mat, (col, row))
                resized = cv2.resize(new_image,(1024, 1024), 0, 0, interpolation=cv2.INTER_NEAREST)
                path=root
                file_new_name = fname[:-4]+str(i)+'.jpg'
                cv2.imwrite(os.path.join(path,file_new_name), resized)

