"""
The Dataset class for reference based data.
"""
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from data.augmentation import flip, rotate
from pathlib import Path
from PIL import Image
from glob import glob

# hyperparams for the affine transform
DEGREES = [-10, 10]
TRANSLATE = [0.1, 0.1]
SCALE_RANGES = [0.8, 1.2]
SHEARS = None
IMG_SIZE = [32, 32]
MODE = T.InterpolationMode.BILINEAR


class RefDataset(Dataset):

    def __init__(self, ipath, augment=False, use_refs=3, blur=None) -> None:
        """
        ipath: Path where the hq and lq images are located.
        augment: Whether to use data augmentation.
        use_refs: Number of refs to use.
        blur: Whether to blur the input [kernel_size, sigma]
        """
        self.ipath = ipath
        self.augment = augment
        self.use_refs = use_refs  # not using this right now
        self.hq_paths = sorted(glob(f"{ipath}/hq/*"))
        self.blur = blur
        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hq_paths)
    
    def __getitem__(self, index):
        hq_path = self.hq_paths[index]
        img_name = Path(hq_path).stem

        # get the hq and lq image
        hq = Image.open(hq_path)
        lq = Image.open(f'{self.ipath}/lq/{img_name}.png')

        if self.augment:
            pr = T.RandomAffine.get_params(degrees=[-5, 5], 
                                        translate=[0.02, 0.02], 
                                        scale_ranges=[0.95, 1.05], 
                                        shears=None, 
                                        img_size=[256, 256])
            hq = F.affine(hq, *pr, interpolation=Image.BICUBIC)
            lq = F.affine(lq, *pr, interpolation=Image.BICUBIC)
        
        # converting to pytorch tensors
        hq = self.transform(hq)
        lq = self.transform(lq)
        if self.blur:
            lq = F.gaussian_blur(lq, self.blur[0], self.blur[1])

        # get the rf image and downsample it
        refs = []
        for i in range(self.use_refs):
            rf = Image.open(f"{self.ipath}/ref/{img_name[:-5]}/{i+1}.png")
            if self.augment:
                rf = F.affine(rf, *pr, interpolation=Image.BICUBIC)
            rf = self.transform(rf)
            refs.append(rf)

        # flips and rotations
        if self.augment:
            hq, lq, refs = flip(hq, lq, refs)
            hq, lq, refs = rotate(hq, lq, refs)

        return hq, lq, refs
