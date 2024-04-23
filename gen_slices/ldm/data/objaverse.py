import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class ObjaverseBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 split,
                 size=None,
                 interpolation="bilinear",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        self.split = split
        self.n_views = 12
        self.img_size = 128
        with open(self.data_paths, "r") as f:
            self.image_ids = f.read().splitlines()
        if self.split == 'trainval_rec': 
            self._length_ = len(self.image_ids)
            self.image_ids = self.image_ids * self.n_views
        self._length = len(self.image_ids)

        self.labels = {
            "file_path_": [l for l in self.image_ids],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def png_2_whitebg(self, img):
        img_rgb = np.array(img)[:, :, 0:3]
        img_alpha = np.array(img)[:, :, 3:4]
        img_alpha_ = (img_alpha == 0).astype(np.float32)
        img_ret = np.ones(img_rgb.shape) * 255 * img_alpha_ + img_rgb * (1 - img_alpha_)
        ret = Image.fromarray(img_ret.astype(np.uint8))
        return ret


    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        if self.split == 'train':
            cmr_angle_idx = random.randint(0, self.n_views - 1)
        elif self.split in ['val', 'test']:
            cmr_angle_idx = 4
        else:
            cmr_angle_idx = i // self._length_
        
        cmr_angle = "%03d"%cmr_angle_idx
        
        img_list = []
        for axis in ['X', 'Z', 'Y']:
            if axis == 'Z':
                slice_list = ['4', '3', '2', '1']
            else:
                slice_list = ['1', '2', '3', '4']

            for part in slice_list:
                img_slice = Image.open(f'{self.data_root}/01_img_slices/{example["file_path_"]}/{cmr_angle}/{axis}_{part}.png')
                img_slice = self.png_2_whitebg(img_slice)
                img_slice = img_slice.resize((self.size, self.size), resample=self.interpolation)
                img_slice = np.array(img_slice)[:, :, :]
                img_list.append(img_slice)
        
        img_ipt_view = Image.open(f'{self.data_root}/00_img_input/{example["file_path_"]}/{cmr_angle}.png')
        img_ipt_view = self.png_2_whitebg(img_ipt_view)
        img_ipt_view_rs = img_ipt_view.resize((self.size, self.size), resample=self.interpolation)
        img_ipt_view = np.array(img_ipt_view_rs)

        img_list.append(img_ipt_view)
        image = np.concatenate(img_list, -1)

        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        example["img_ipt_view"] = (img_ipt_view / 127.5 - 1.0).astype(np.float32)

        example["segmentation"] = np.zeros((3, 32, 32)).astype(np.float32)

        return example


class ObjaverseTrain(ObjaverseBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="../data/objaverse/03_splits/train.lst", data_root="../data/objaverse", split='train', **kwargs)


class ObjaverseValidation(ObjaverseBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="../data/objaverse/03_splits/val.lst", data_root="../data/objaverse", split='val',
                         flip_p=flip_p, **kwargs)

class ObjaverseTest(ObjaverseBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="../data/objaverse/03_splits/test.lst", data_root="../data/objaverse", split='test',
                         flip_p=flip_p, **kwargs)

class ObjaverseTrainValRec(ObjaverseBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="../data/objaverse/03_splits/trainval.lst", data_root="../data/objaverse", split='trainval_rec',
                         flip_p=flip_p, **kwargs)