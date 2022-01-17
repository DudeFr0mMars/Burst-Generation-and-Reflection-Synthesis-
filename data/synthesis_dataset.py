import os.path
from data.image_folder import make_dataset
import torch.utils.data as data
from PIL import Image
import random
import torchvision.transforms as transforms
import cv2
import numpy as np


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

class SynthesisDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.loadSizeH = opt.loadSizeH
        self.loadSizeW = opt.loadSizeW
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)            

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
            
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if opt.phase == 'train':
            self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C')
            self.C_paths = make_dataset(self.dir_C)
            self.C_paths = sorted(self.C_paths)
            self.C_size = len(self.C_paths)

    def get_resized(self, img):

        transform_list = []
        transform_list.append(transforms.Resize((self.loadSizeH, self.loadSizeW)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        img = transform(img)

        return img

    def get_crop_resized(self, img):

        transform_list = []
        transform_list.append(transforms.CenterCrop((img.size[1]-img.size[1]%self.loadSizeH, img.size[0]-img.size[0]%self.loadSizeW)))
        transform_list.append(transforms.Resize((self.loadSizeH, self.loadSizeW)))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(transform_list)
        img = transform(img)

        return img

    def __getitem__(self, index):
        index_A = random.randint(0, self.A_size - 1)
        A_path = self.A_paths[index_A]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]            

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_img_origin = A_img

        if self.opt.phase == 'train':
            type_list = ['focused','defocused','ghosting']
            reflection_type = type_list[random.randint(0, 2)]
        elif self.opt.phase == 'test':
            reflection_type = self.opt.type

        # for different reflection types
        # Focused reflection
        if reflection_type == 'focused':
            A_img = np.asarray(A_img)
            B_img = np.asarray(B_img)
            # Generate blend masks, here: linear, horizontal fading from 1 to 0.6 and from 0 to 0.4
            mask1 = np.repeat(np.tile(np.linspace(1, 0.6, A_img.shape[1]), (A_img.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
            mask2 = np.repeat(np.tile(np.linspace(0, 0.4, B_img.shape[1]), (A_img.shape[0], 1))[:, :, np.newaxis], 3, axis=2)
            A_img = A_img * mask1 + B_img * mask2
            # Generate output by linear blending
            A_img = Image.fromarray(A_img.astype(np.uint8))
            B_img = Image.fromarray(B_img.astype(np.uint8))

        # Defocused reflection
        elif reflection_type == 'defocused':
            A_img = np.asarray(A_img)
            A_img = cv2.GaussianBlur(A_img,(5,5),3,3,0)
            A_img = Image.fromarray(A_img.astype(np.uint8))
        # Ghosting reflection
        elif reflection_type == 'ghosting':
            A_img = np.asarray(A_img)
            shift_x = np.random.randint(20, 40)
            shift_y = np.random.randint(20, 40)
            rows, cols, channels = A_img.shape
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            A_img_shifted = cv2.warpAffine(A_img, M, (cols, rows))

            attenuation = np.random.uniform(0.5, 1)
            A_img = A_img * attenuation + A_img_shifted * (1-attenuation)
            A_img = Image.fromarray(A_img.astype(np.uint8))

            A_img = transforms.functional.crop(A_img, shift_y, shift_x, cols-shift_y, rows-shift_x)
            A_img = A_img.resize((rows, cols), Image.BILINEAR)
        
        A = self.get_crop_resized(A_img)
        A_origin = self.get_crop_resized(A_img_origin)
        B = self.get_resized(B_img)

        if self.opt.phase == 'train':
            index_C = random.randint(0, self.C_size - 1)
            C_path = self.C_paths[index_C]
            C_img = Image.open(C_path).convert('RGB')
            C = self.get_resized(C_img)

        if self.opt.phase == 'train':
            return {'A': A, 'A_origin': A_origin, 'B': B, 'C': C,
                    'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path}
        else:
            return {'A': A, 'A_origin': A_origin, 'B': B,
                    'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'SynthesisDataset'
