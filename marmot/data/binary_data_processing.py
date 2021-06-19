import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TextImageDatasetBinary(Dataset):
    def __init__(self, data, docid_varname, text_varname, img_varname, caption_varname, label_varname,
                imgs_dir='/', img_filler='foo.jpg', test=False, transform=None, cleanup=False, byte_min_cleanup=1000, windows=False):
        self.test = test
        self.label_varname = label_varname
        self.data = data
        self.windows = windows
        if not windows:
            self.imgs_dir = Path(imgs_dir)
        else:
            self.imgs_dir = imgs_dir

        self.transform = transform

        # Check for missing images
        self.data['pic'] = 1
        for i in range(len(self.data)):
            if windows:
                img_path = Path(self.imgs_dir + '/' + str(self.data.loc[i, img_varname]))
            else:
                img_path = self.imgs_dir/self.data.loc[i, img_varname]
            # Check to see if file paths exists
            if not os.path.isfile(img_path):
                self.data.loc[i, img_varname] = img_filler
                self.data.loc[i, 'pic'] = 0
                continue

        # Clean up if there are messy images
        if cleanup:
            for i in range(len(self.data)):
                if windows:
                    img_path = Path(self.imgs_dir + '/' + str(self.data.loc[i, img_varname]))
                else:
                    img_path = self.imgs_dir/self.data.loc[i, img_varname]
                if self.IsNaN(self.data.loc[i, img_varname]):
                    self.data.loc[i, img_varname] = img_filler
                    self.data.loc[i, 'pic'] = 0
                    continue
                # Check to see if they are not less than some given minimum size
                if os.path.getsize(img_path) < byte_min_cleanup:
                    self.data.loc[i,'picfiles'] = img_filler
                    self.data.loc[i, 'pic'] = 0



        self.doc_id = self.data[docid_varname].values
        self.text = self.data[text_varname].values
        self.image = self.data[img_varname].values
        self.image_caption = self.data[caption_varname].values
        self.pic = self.data['pic'].values

        if not test:
            self.label = self.data[label_varname].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """Read in Doc ID"""
        doc_id = self.doc_id[idx]

        """Read in Text"""
        text = self.text[idx]

        """Read in Image"""
        if windows:
            img_name = Path(self.imgs_dir + '/' + str(self.image[idx]))
        else:
            img_name = self.imgs_dir/self.image[idx]
        image_raw = Image.open(img_name).convert('RGB')
        image_raw = np.asarray(image_raw)
        image = self.transform(image_raw)

        """Read in Image Caption"""
        image_caption = self.image_caption[idx]

        """Read in Pic Status"""
        pic = self.pic[idx]

        """Read in Labels"""
        if not self.test:
            label = self.label[idx]

        if not self.test:
            sample = {'id': doc_id, 'image': image, 'image_caption': image_caption, 'text': text, 'pic': pic, 'label': label}
        else:
            sample = {'id': doc_id, 'image': image, 'image_caption': image_caption, 'pic': pic, 'text': text}

        return sample

    def IsNaN(self, string):
        return string != string
