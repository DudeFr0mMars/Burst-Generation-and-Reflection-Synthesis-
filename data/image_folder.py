import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
import os
import os.path
os.mkdir('horiz-a')
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    angle1=0.2
    angle2=0.275
    angle3=0.35
    angle4=0.425
    angle5=0.5
    angle6=0.575
    angle7=0.65
    angle_list=[angle1,angle2,angle3,angle4,angle5,angle6,angle7]

    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                for burst in range(7):
                    img = cv2.imread(path)
                    zoom_factor=1
                    rows, cols, ch = img.shape
                    pts1 = np.float32([[cols*.25, rows*.95],[cols*.90, rows*.95],[cols*.10, 0],[cols,0]])
                    pts2 = np.float32([[cols*0.1, rows],[cols,rows],[0,0],[cols,0]])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    dst = cv2.warpPerspective(img, M, (cols, rows))
                    angle_size=angle_list[burst]
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

                    resized = cv2.resize(dst,(1024, 1024), 0, 0, interpolation=cv2.INTER_NEAREST)
                    path_burst='horiz-a'
                    cv2.imwrite(os.path.join(path_burst, f'{burst}-{path}'), resized)
                    images.append(os.path.join(path_burst, f'A-{i}.jpg'))

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
