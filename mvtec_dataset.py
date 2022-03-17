"""
This is an example data loader for pytorch which uses the datatool API to load the datatool output JSON and read samples.
The example task prepares a transform function of the images and segmentation masks into 300x300 png images and offers
a getitem function to load one image, associated annotations information, and associated masks if the image is a defect
image.

Input:
The input is a datatool output directory containing the dataset.json and samples directory

"""

from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import os
import sys
import inspect
import random

cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(cur_path, '..'))
from datatool_api.config.APIConfig import DTAPIConfig
from datatool_api.models.DTDataset import DTDataset
from custom_dataset_model import DTDatasetCustom

CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, root_path: str, class_names_list=['bottle'], is_train=True,
                 resize=256, cropsize=224, operating_mode: str = 'memory'):
        """
        Instantiate the dataset instance

        :param data_dir: Directory containing the datatool output which contains the "sample_files" directory and
        "dataset.json" file
        :param operating_mode: Operating mode for the datatool api to handle the dataset, based on the data size,
        user can choose [memory, disk or ramdisk]

        """
        for class_name in class_names_list:
            assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)

        Dataset.__init__(self)
        self.data_dir = root_path
        self.operating_mode = operating_mode
        # self.kwargs = kwargs
        self.class_names_list = class_names_list
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # No need to validate when loading back the API output
        DTAPIConfig.disable_validation()
        # Load the dataset.json using the datatool API
        self.dataset = DTDatasetCustom(name='input_dataset',
                                       operatingMode=self.operating_mode).load_from_json(os.path.join(self.data_dir,
                                                                                                      'dataset.json'),
                                                                                         element_list=['images',
                                                                                                       'counters',
                                                                                                       'annotations'])

        # Example Transform to resize and normalize the images: Modify it to write a more complex transform
        # self.transform = transforms.Compose([transforms.Resize(self.kwargs['model_input_size']),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize(mean=self.kwargs['normalization']['mean'],
        #                                                          std=self.kwargs['normalization']['std'])])
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      # T.CenterCrop(cropsize),
                                      T.ToTensor()])
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #            std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         # T.CenterCrop(cropsize),
                                         T.ToTensor()])

        self.custom_index_list_class_based_good = []
        self.custom_index_list_class_based_defect = []
        self.custom_index_list_class_based = []
        i = 0
        while i < self.dataset.counters.imagesCounter:
            if self.dataset.annotations.get(str(i)).objectCategory == 'good' and self.dataset.annotations.get(
                    str(i)).objectType in self.class_names_list:
                self.custom_index_list_class_based_good.append(i)
                self.custom_index_list_class_based.append(i)
            elif self.dataset.annotations.get(str(i)).objectCategory == 'defect' and self.dataset.annotations.get(
                    str(i)).objectType in self.class_names_list:
                self.custom_index_list_class_based_defect.append(i)
                self.custom_index_list_class_based.append(i)

        # Add any extra needed variables here
        # TODO - Implemented by the datatool creator

    def __len__(self):
        # TODO - Implemented by the datatool creator
        return len(self.custom_index_list_class_based)

    def __getitem__(self, index):
        # Return
        image_name = self.dataset.images.get(str(self.custom_index_list_class_based[index])).id + '.png'
        img = Image.open(os.path.join(self.data_dir, 'sample_files', image_name)).convert('RGB')
        img = self.transform(img)
        if self.dataset.annotations.get(str(self.custom_index_list_class_based[index])).objectCategory == 'good':
            #           return img, self.dataset.annotations.get(index)
            return img, 0
        else:
            # segmask_name = self.dataset.annotations.get(index).id + '_mask.png'
            # segmask = Image.open(os.path.join(self.data_dir, 'sample_files', segmask_name ))
            # segmask = self.transform(segmask)
            return img, 1


class BDDataset(Dataset):
    def __init__(self,
                 root_path='../data',
                 bdd_folder_path='mvtec',
                 class_names_list=['bottle'],
                 is_train=True,
                 resize=256,
                 cropsize=224,
                 ways=2,
                 shots=3,
                 query=5, operating_mode: str = 'memory'):

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

        DTAPIConfig.disable_validation()
        # Load the dataset.json using the datatool API
        self.dataset = DTDatasetCustom(name='input_dataset',
                                       operatingMode=self.operating_mode).load_from_json(os.path.join(self.data_dir,
                                                                                                      'dataset.json'),
                                                                                         element_list=['images',
                                                                                                       'counters',
                                                                                                       'annotations'])

        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      # T.CenterCrop(cropsize),
                                      T.ToTensor()])
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #            std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         # T.CenterCrop(cropsize),
                                         T.ToTensor()])

        self.custom_index_list_class_based_good = []
        self.custom_index_list_class_based_defect = []
        self.custom_index_list_class_based = []
        i = 0
        while i < self.dataset.counters.imagesCounter:
            if self.dataset.annotations.get(str(i)).objectCategory == 'good' and self.dataset.annotations.get(
                    str(i)).objectType in self.class_names_list:
                self.custom_index_list_class_based_good.append(i)
                self.custom_index_list_class_based.append(i)
            elif self.dataset.annotations.get(str(i)).objectCategory == 'defect' and self.dataset.annotations.get(
                    str(i)).objectType in self.class_names_list:
                self.custom_index_list_class_based_defect.append(i)
                self.custom_index_list_class_based.append(i)

        assert len(self.custom_index_list_class_based_good) > 0
        assert len(self.custom_index_list_class_based_defect) > self.shots + self.query

    def __len__(self):
        # TODO - Implemented by the datatool creator
        return len(self.custom_index_list_class_based)

    def __getitem__(self, idx):
        batch_x, batch_y = [], []

        # defect
        for i in range((self.shots + self.query)):
            image_name = self.dataset.images.get(
                str(self.custom_index_list_class_based_defect[idx * (self.shots + self.query) + i])).id + '.png'
            img = Image.open(os.path.join(self.data_dir, 'sample_files', image_name)).convert('RGB')
            img = self.transform(img)

            batch_x.append(img)
            batch_y.append(1)

        # good
        for _ in range((self.shots + self.query)):
            rand_i = random.randint(0, len(self.custom_index_list_class_based_good))
            image_name = self.dataset.images.get(str(self.custom_index_list_class_based_defect[rand_i])).id + '.png'
            img = Image.open(os.path.join(self.data_dir, 'sample_files', image_name)).convert('RGB')
            img = self.transform(img)

            batch_x.append(img)
            batch_y.append(0)

        return torch.stack(batch_x), torch.tensor(batch_y)


def main():
    params = {
        'image_type': 'RGB',  # Image type that the model is going to take as input, in this case it is
        # 3 channel RGB
        'model_input_size': (256, 256),  # Width x Height of input tensor for the model
        # 'min_landmark_count': 17,  # Min no of 2d landmarks which must be present for a subject to include it
        # in loading candidate list
        'normalization': {  # Normalization parameters for image
            'mean': [154.78, 118.57, 101.74],
            'std': [53.80, 48.19, 46.15]
        }
    }

    # TODO: Set by the user
    data_dir = '<Input directory containing the samples and dataset.json>'

    # Create dataset instance
    dataset = MVTecDataset(data_dir=data_dir, operating_mode='memory', **params)
    for i in range(len(dataset)):
        print(dataset.__getitem__(i))


if __name__ == main():
    main()
