import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
               'dagm_c1', 'dagm_c2', 'dagm_c3', 'dagm_c4', 'dagm_c5', 'dagm_c6', 'kolectorsdd2_train']


class MVTecDataset(Dataset):
    def __init__(self, root_path='../data', class_names_list=['bottle'], is_train=True,
                 resize=256, cropsize=224):
        for class_name in class_names_list:
            assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.root_path = root_path
        self.class_names_list = class_names_list
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.mvtec_folder_path = os.path.join(root_path, 'mvtech_cleaned')

        # download dataset if not exist
        # self.download()

        # load dataset
        # self.x, self.y, self.mask = self.load_dataset_folder()
        self.x, self.y = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      #T.CenterCrop(cropsize),
                                      T.ToTensor()])
                                      #T.Normalize(mean=[0.485, 0.456, 0.406],
                                      #            std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         #T.CenterCrop(cropsize),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        # x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        # if y == 0:
        #     mask = torch.zeros([1, self.cropsize, self.cropsize])
        # else:
        #     mask = Image.open(mask)
        #     mask = self.transform_mask(mask)

        return x, y  #, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y = [], []

        for class_name in self.class_names_list:
            img_dir = os.path.join(self.mvtec_folder_path, class_name, phase)
            # gt_dir = os.path.join(self.mvtec_folder_path, class_name, 'ground_truth')

            img_types = sorted(os.listdir(img_dir))
            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)
                                         if f.endswith('.png')])
                x.extend(img_fpath_list)

                # load gt labels
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    # mask.extend([None] * len(img_fpath_list))
                else:
                    y.extend([1] * len(img_fpath_list))
                    # gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    # gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                    #                  for img_fname in img_fname_list]
                    # mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y) #  , list(mask)

    def download(self):
        """Download dataset if not exist"""

        if not os.path.exists(self.mvtec_folder_path):
            tar_file_path = self.mvtec_folder_path + '.tar.xz'
            if not os.path.exists(tar_file_path):
                download_url(URL, tar_file_path)
            print('unzip downloaded dataset: %s' % tar_file_path)
            tar = tarfile.open(tar_file_path, 'r:xz')
            tar.extractall(self.mvtec_folder_path)
            tar.close()

        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class BDDDataset(Dataset):
    def __init__(self,
                 root_path='../data',
                 bdd_folder_path='mvtec',
                 class_names_list=['bottle'],
                 is_train=True,
                 resize=256,
                 cropsize=224,
                 ways=2,
                 shots=3,
                 query=5):

        for class_name in class_names_list:
            assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        self.root_path = root_path
        self.class_names_list = class_names_list
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.dataset_folder_path = os.path.join(root_path, bdd_folder_path)
        self.ways = ways
        self.shots = shots
        self.query = query

        # load dataset
        self.x_0, self.y_0, self.x_1, self.y_1 = self.load_dataset_folder()
        assert len(self.y_1) > self.shots + self.query
        assert len(self.y_0) > 0
        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.ToTensor()])
                                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                                    #            std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.ToTensor()])

    def __getitem__(self, idx):
        batch_x, batch_y = [], []

        # defects
        for i in range((self.shots + self.query)):
            x, y = self.x_1[idx * (self.shots + self.query) + i], self.y_1[idx * (self.shots + self.query) + i]
            x = Image.open(x).convert('RGB')
            x = self.transform_x(x)

            batch_x.append(x)
            batch_y.append(y)

        # good
        for _ in range((self.shots + self.query)):
            rand_i = random.randint(0, len(self.y_0) - 1)
            x, y = self.x_0[rand_i], self.y_0[rand_i]
            x = Image.open(x).convert('RGB')
            x = self.transform_x(x)
            batch_x.append(x)
            batch_y.append(y)

        return torch.stack(batch_x), torch.tensor(batch_y)

    def __len__(self):
        return len(self.y_1) // (self.shots + self.query)

    def load_dataset_folder(self):
        # phase = 'train' if self.is_train else 'test'
        phase = 'test'  # if self.is_train else 'test'
        x_0, x_1, y_0, y_1 = [], [], [], []

        for class_name in self.class_names_list:
            img_dir = os.path.join(self.dataset_folder_path, class_name, phase)
            img_types = sorted(os.listdir(img_dir))

            for img_type in img_types:

                # load images
                img_type_dir = os.path.join(img_dir, img_type)
                if not os.path.isdir(img_type_dir):
                    continue
                img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                         for f in os.listdir(img_type_dir)
                                         if f.endswith('.png')])

                # load gt labels
                if img_type == 'good':
                    x_0.extend(img_fpath_list)
                    y_0.extend([0] * len(img_fpath_list))
                else:
                    x_1.extend(img_fpath_list)
                    y_1.extend([1] * len(img_fpath_list))

        assert len(x_0) == len(y_0), 'number of x and y should be same'
        assert len(x_1) == len(y_1), 'number of x and y should be same'

        return list(x_0), list(y_0), list(x_1), list(y_1)

