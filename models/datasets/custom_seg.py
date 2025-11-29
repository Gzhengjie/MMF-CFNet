import ctypes
import glob
import os
import os.path as osp
from collections import OrderedDict
from functools import reduce
import gdal
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from numpy import ndarray
from torch.utils.data import Dataset
from .transforms.albu import ToTensorTest

# libbytiff = ctypes.CDLL(r"D:\Users\szt\Anaconda3\envs\pytorch\DLLs\libtiff.dll")
# libbytiff.TIFFSetWarningHandler.argtypes = [ctypes.c_void_p]
# libbytiff.TIFFSetWarningHandler.restype = ctypes.c_void_p
# libbytiff.TIFFSetWarningHandler(None)

class CustomSegDataset(Dataset):
    """Custom datasets for change detection. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── my_dataset
        │   │   ├── train
        │   │   │   ├── img1_dir
        │   │   │   ├── img1_dir
        │   │   │   ├── label_dir
        │   │   ├── val
        │   │   │   ├── img1_dir
        │   │   │   ├── img1_dir
        │   │   │   ├── label_dir

    The imgs/gt pair of CustomDataset should be of the same except suffix.
    A valid imgs/gt filename pair should be like ``xxx{img_suffix}`` and
    ``xxx{seg_map_suffix}`` (extension is also included in the suffix).
    Args:
        img_dir (str): Path to image directory.
        sub_dir_1 (str): Path to the directory of the first temporal images.
            e.g. 'A' in LEVIR-CD dataset (LEVIR-CD/train/A). Default: 'A'
        sub_dir_2 (str): Path to the directory of the second temporal images.
            e.g. 'B' in LEVIR-CD dataset (LEVIR-CD/train/B). Default: 'B'
        ann_dir (str): Path to ground truth directory.
        img_suffix (str): Suffix of images. Default: '.jpg'
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str|None): Split txt file. If split is specified, only file
            with suffix in the splits will be loaded. Otherwise, all images
            in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): Whether to the test mode.
        size (int): The size of input images.
        debug (bool): Whether to use debug mode. i.e. visualization.
    """

    def __init__(self,
                 img_dir=None,
                 ann_dir=None,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 transform=None,
                 split=None,
                 data_root=None,
                 test_mode=False,
                 size=256,
                 debug=False):
        self.transform = transform
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.size = size
        self.debug = debug

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)

        # load annotations
        self.img_infos = self.load_infos(self.img_dir, self.img_suffix,
                                         self.seg_map_suffix,  self.ann_dir,
                                         self.split)

        # transform/augment data
        if self.transform is None:
            self.transform = self.get_default_transform() if not self.test_mode \
                else self.get_test_transform()

        # debug, visualize augmentations
        if self.debug:
            self.transform = A.Compose([t for t in self.transform if not isinstance(t, (# A.Normalize,
                                                                                        ToTensorV2,
                                                                                        ToTensorTest))])

    def load_infos(self, img_dir, img_suffix, seg_map_suffix, ann_dir, split):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of datasets.
            list的大小是数据量，dict包括basename基础文件名、img影像路径、ann标签路径
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip() #删除文件名前缀和后缀中的空白
                    img_info = dict(filename=img_name)
                    img_info['img'] = dict(img1_path=osp.join(img_dir, sub_dir_1, img_name),
                                           img2_path=osp.join(img_dir, sub_dir_2, img_name))
                    if ann_dir is not None:
                        seg_map_path = osp.join(ann_dir,
                                           img_name.replace(img_suffix, seg_map_suffix))
                        img_info['ann'] = dict(ann_path=seg_map_path)
                    img_infos.append(img_info)
        else:
            for img in glob.glob(osp.join(img_dir, '*'+img_suffix)):
                #加载影像
                img_name = osp.basename(img) #取不带路径的名字
                img_info = dict(filename=img_name)
                img_info['img'] = dict(img_path=osp.join(img_dir, img_name))

                #加载标签
                if ann_dir is not None:
                    seg_map_path = osp.join(ann_dir,
                                            img_name.replace(img_suffix, seg_map_suffix))
                    img_info['ann'] = dict(ann_path=seg_map_path)
                img_infos.append(img_info)

        print(f'Loaded {len(img_infos)} images')
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def get_default_transform(self):
        """Set the default transformation."""

        default_transform = A.Compose([
            A.Resize(self.size, self.size),
            # A.Normalize(),
            ToTensorV2()
        ])
        return default_transform

    def get_test_transform(self):
        """Set the test transformation."""
        pass

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def get_image(self, img_info):
        """Open and read the image.
        Args:
            img_info (dict): a dict with img info.
        Returns:
            dict: image info with new keys.
        """

        # img1 = cv2.cvtColor(cv2.imread(img_info['img']['img1_path']), cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(cv2.imread(img_info['img']['img2_path']), cv2.COLOR_BGR2RGB)

        # tif1 = TIFF.open(img_info['img']['img1_path'], mode="r")
        # img1 = tif1.read_image()
        # tif2 = TIFF.open(img_info['img']['img2_path'], mode="r")
        # img2 = tif2.read_image()

        tif = gdal.Open(img_info['img']['img_path'])
        img = tif.ReadAsArray()
        # img = img[4:10,:,:]
        img = img.astype(np.float32)
        img = self.get_ZS(img)
        # print("---------------------------------------------------------")
        # print(img.shape)
        img = img.transpose(1, 2, 0) #数组转置 3*224*224 => 224*224*3

        # tif2 = gdal.Open(img_info['img']['img2_path'])
        # img2 = tif2.ReadAsArray()
        # img2 = img2.transpose(1, 2, 0)
        img[np.isnan(img)] = 0

        return img

    # 获取指数
    def get_ZS(self, img):
        # 忽略现了0除以0的情况造成的
        np.seterr(divide='ignore', invalid='ignore')
        blue = img[0, :, :]
        green = img[1, :, :]
        rad = img[2, :, :]
        nir = img[3, :, :]
        swir1 = img[4, :, :]
        swir2 = img[5, :, :]

        NDVI = (rad - nir) / (rad + nir) + 1
        NDSI = (green - swir1) / (swir1 + green) + 1
        NDWI = (green - nir) / (green + nir) + 1

        img = np.append(img, np.array([NDVI]), axis=0)
        img = np.append(img, np.array([NDSI]), axis=0)
        img = np.append(img, np.array([NDWI]), axis=0)

        return img
        # img[np.isnan(img)] = 0
        # for i in range(img.shape[0]):
        #     d = self.normalization(img[i])
        #     img = np.append(img, np.array([d]), axis=0)
        #
        # img[np.isnan(img)] = 0
        # return img[13:].astype(np.float32)

    def get_gt_seg_maps(self, img_info, vis=False):
        """Open and read the ground truth.
        Args:
            img_info (dict): a dict with ann info.
            vis (bool): Whether to use visualization (debug mode).
        Returns:
            dict: ann info with new keys.
        """

        # ann = cv2.imread(img_info['ann']['ann_path'], cv2.IMREAD_GRAYSCALE)
        # ann = ann / 255 if not vis else ann

        # tifa = TIFF.open(img_info['ann']['ann_path'], mode="r")
        # ann = tifa.read_image()

        tifa = gdal.Open(img_info['ann']['ann_path'])
        # print(img_info['ann']['ann_path'])
        ann = tifa.ReadAsArray()

        return ann

    def prepare_img(self, idx):
        """Get image after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Image after pipeline with new keys introduced by pipeline.
        """

        img_info = self.img_infos[idx] # 返回字典 包括basename基础文件名、img影像路径、ann标签路径
        img = self.get_image(img_info)
        return img, img_info['filename']

    def prepare_img_ann(self, idx):
        """Get image and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Image and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        img = self.get_image(img_info)
        ann = self.get_gt_seg_maps(img_info, self.debug)
        return img, ann, img_info['filename']

    def format_results(self, results, **kwargs):
        """Place holder to format result to datasets specific output."""
        pass

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if ann_dir is not None).
        """
        raise NotImplementedError

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)