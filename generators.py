# import packages
import SimpleITK as sitk
import numpy as np
import os
import random
import torch.utils.data
from skimage import exposure
from skimage import transform
from skimage.filters import unsharp_mask, gaussian
import torch

def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device, dtype=torch.float)
    return x


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.detach().numpy()
    return x


# CAMUS dataset
class CamusPairDataset(torch.utils.data.Dataset):
    # Return img + mask (ED + ES)
    def __init__(self, files_list, img_path, imgsize=128, transform=None):
        super(CamusPairDataset, self).__init__()
        self.files = files_list
        self.images_path = img_path
        self.transform = transform  # flag
        self.imgsize = imgsize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patient = self.files[idx]
        # dist_ed/es is already scaled ~
        img_ed, mask_ed, img_es, mask_es = self.load(patient)
        sampling = 1
        if self.transform:
            rand_seq_pair = create_rand_seq([self.imgsize, self.imgsize])
            img_ed, mask_ed, sampling = self.transform((img_ed, mask_ed), size = [self.imgsize, self.imgsize], rand_seq = rand_seq_pair)
            img_es, mask_es, sampling = self.transform((img_es, mask_es), size = [self.imgsize, self.imgsize], rand_seq = rand_seq_pair)
        img_ed = np.expand_dims(img_ed, 0)
        img_es = np.expand_dims(img_es, 0)
        zeros = np.zeros((img_ed.shape))
        return patient[0], patient[1], [img_es, img_ed], mask_es, [img_ed, img_es, zeros], mask_ed, sampling

    def load(self, ID):
        i, ch = ID
        i = int(i)
        ch = int(ch)
        id_pre = 'patient' + str(i).zfill(4) + '/patient' + str(i).zfill(4) + '_' + str(int(ch)) +'CH_'
        patient_path_ed = os.path.join(self.images_path, id_pre) + 'ED.mhd'
        img_ed = sitk.GetArrayFromImage(sitk.ReadImage(patient_path_ed))
        patient_path_es = os.path.join(self.images_path, id_pre) + 'ES.mhd'
        img_es = sitk.GetArrayFromImage(sitk.ReadImage(patient_path_es))
        try:
            mask_path_ed = os.path.join(self.images_path, id_pre) + 'ED_gt.mhd'
            mask_ed = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_ed))
            mask_ed = mask_ed[0].astype(np.int16)

            mask_path_es = os.path.join(self.images_path, id_pre) + 'ES_gt.mhd'
            mask_es = sitk.GetArrayFromImage(sitk.ReadImage(mask_path_es))
            mask_es = mask_es[0].astype(np.int16)
            mask_ed = np.rollaxis(np.eye(4, dtype=np.uint8)[mask_ed], -1, 0)
            mask_es = np.rollaxis(np.eye(4, dtype=np.uint8)[mask_es], -1, 0)
        except:
            mask_ed = np.zeros((4, img_ed.shape[1], img_ed.shape[2]))
            mask_es = np.zeros((4, img_es.shape[1], img_es.shape[2]))
        return img_ed, mask_ed, img_es, mask_es


# ==================================================================================
# functions for data transformation (augmentation)
# Possible tranforms:
# 1. image quality: sharpening, blurring, noise
# 2. image appearance: brightness, contrast
# 3. spatial transforms: rotation, scaling
# ==================================================================================
def zoom(sample, size = [128,128], rand_seq = None):
    img, mask = sample
    # zoom
    if img is not None:
        k, y, x = img.shape
        resolution = [y / size[0] * 0.15, x / size[1] * 0.3]
        k = img.shape[0]
        img = transform.resize(img, (k, size[0], size[1]), 1)
    if mask is not None:
        k, y, x = mask.shape
        resolution = [y / size[0] * 0.15, x / size[1] * 0.3]
        mask = np.round(transform.resize(mask.astype(float), (k, size[0], size[1]), 1))
    return img, mask, resolution


def crop_resize(img, new_l, resol, limit=256):
    k, y, x = img.shape
    resolution = [y / new_l * resol[0], x / new_l * resol[1]]
    img = transform.resize(img, (k, new_l, new_l), 1)
    if new_l > limit:
        s = int((new_l - limit) / 2)
        img = img[:, s:(s + limit), s:(s + limit)]
    else:
        s1 = int((limit - new_l) / 2)
        s2 = limit - new_l - s1
        img = np.pad(img, ((0, 0), (s1, s2), (s1, s2)), mode='edge')
    return img, resolution


def rotation(img, angle):
    new_out = []
    for im in img:
        new_out.append(transform.rotate(im, angle,mode='edge'))
    return np.array(new_out)


def create_rand_seq(size):
    rand_seq = []
    rand_seq.append((random.random(), random.randint(-15, 15)))
    rand_seq.append((random.random(), random.random()))
    rand_seq.append((random.random(), random.random()))
    rand_seq.append((random.random(), random.random(), random.random()))
    rand_seq.append((random.random(), random.random(), random.random()))
    rand_seq.append((random.random(), random.random()))
    new_rand = random.random()
    std = new_rand * 0.09 + 0.01
    rand_seq.append((random.random(), new_rand, np.random.normal(0, std, (size[0], size[1]))))
    return rand_seq


def data_aug(sample, size=[128, 128], prob=0.5, rand_seq = None):
    img, mask = sample
    if rand_seq is None:
        rand_seq = create_rand_seq(size)
    # zoom
    if img is not None:
        k, y, x = img.shape
        resolution = [y / size[0] * 0.15, x / size[1] * 0.3]
        img = transform.resize(img, (k, size[0], size[1]), 1)
    if mask is not None:
        k, y, x = mask.shape
        mask = np.round(transform.resize(mask.astype(float), (k, size[0], size[1]), 1))

    # rotation [-10,10]
    if rand_seq[0][0] > prob:
        angle = rand_seq[0][1]
        if img is not None:
            img = rotation(img, angle)
        if mask is not None:
            mask = rotation(mask, angle)
    # crop and scale [0.7-1.3]
    if rand_seq[1][0] > prob:
        scale = rand_seq[1][1] * 0.6 + 0.7
        new_l = int(scale * size[0])
        if img is not None:
            img, resolution = crop_resize(img, new_l, resolution, limit=size[0])
        if mask is not None:
            mask, _ = crop_resize(mask, new_l, resolution, limit=size[0])


    if img is not None:
        new_img = []
        # no operation on mask
        k, y, x = img.shape
        for f in range(k):
            img0 = img[f]

            # brightness [0.9,1.1]
            if rand_seq[2][0] > prob:
                # image: I * gamma
                bright = rand_seq[2][1] * 0.2 + 0.9
                img0 = exposure.adjust_gamma(img0, bright)

            # contrast [0.8,1.2]
            if rand_seq[3][0] > prob:
                # cutoff [0.4,0.6]
                cutoff = rand_seq[3][1]* 0.2 + 0.4
                # gain [4,10]
                gain = rand_seq[3][2] * 6 + 4
                img0 = exposure.adjust_sigmoid(img0, cutoff=cutoff, gain=gain)

            # sharpening: std [0.25-1.5] amount [1,2]
            if rand_seq[4][0] > prob:
                std = rand_seq[4][1] * 1.25 + 0.25
                amount = rand_seq[4][2] * 1 + 1
                img0 = unsharp_mask(img0, radius=std, amount=amount)

            # blurring std[0.25,1.5]
            if rand_seq[5][0] > prob:
                std = rand_seq[5][1]* 1.25 + 0.25
                img0 = gaussian(img0, sigma=std)

            # noise : speckle noise
            if rand_seq[6][0] > prob:
                # gaussian noise std [0.01,0.1]
                std = rand_seq[6][1]* 0.09 + 0.01
                gauss = rand_seq[6][2]
                gauss = np.sqrt(img0) * gauss
                img0 = gauss + img0
            new_img.append(img0)
        img = np.array(new_img)

    return img, mask, resolution