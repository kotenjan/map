from torchvision import transforms
from PIL import Image
import glob
from torch.utils.data import Dataset

class SatelliteToMapDataset(Dataset):
    def __init__(self, root_dir, resize=None, augmentation=True):
        self.file_list = glob.glob(root_dir + '/*.jpg')
        self.transform = self.get_transforms(resize, augmentation)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        w, h = img.size

        img_A = img.crop((0, 0, w / 2, h))  # Satellite image
        img_B = img.crop((w / 2, 0, w, h))  # Map image

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {'satellite_image': img_A, 'map_image': img_B}
    
    def get_transforms(self, resize=None, augmentation=True):
        transform_list = []

        if resize is not None:
            transform_list.append(transforms.Resize(resize))

        transform_list.append(transforms.ToTensor())

        if augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.4),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
            ])

        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        return transforms.Compose(transform_list)
