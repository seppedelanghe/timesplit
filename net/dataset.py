import torch, json
import torchvision.transforms as transforms
from lib.models import Annotation

class ImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

class TDADataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str, img_dir: str):
        self.img_dir = img_dir
        self.data = self._read_labels(filepath)

        self.t = ImageCompose([
            transforms.Resize((32, 24)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.data)

    def _read_labels(self, path: str):
        with open(path, 'r') as f:
            data = json.loads(f.read())
            return [Annotation.parse_obj(x) for x in data]

    def __getitem__(self, index):
        item = self.data[index]
        frm, to = item.read(self.img_dir)

        if self.t:
            frm = self.t(frm)
            to = self.t(to)

        return frm, to, item.tda.as_tensor()
