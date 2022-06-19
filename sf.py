import torch, os, json
from PIL import Image
import torchvision.transforms as transforms

class ImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

class SFDataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        self.img_dir = os.path.join(filepath.replace(os.path.basename(filepath), ''), 'images')
        self.labels = self._read_labels(filepath)

        self.t = ImageCompose([
            transforms.Resize((32, 24)), 
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.labels)

    def _read_labels(self, path: str):
        with open(path, 'r') as f:
            return json.loads(f.read())

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.labels[index]['image'])
        image = Image.open(img_path)

        if self.t:
            image = self.t(image)

        return image
