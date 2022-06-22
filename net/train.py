import os

from model import TimeModel
from loss import TimeSplitLoss
from sf import TDADataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

TEST_DB = os.path.join('D:/Projects/eleven/timesplit/annotator/static/data', "db.json")

IN_CNN = [
    # Tuple: (kernel_size, n_filters, stride, padding)
    (7, 64, 2, 3),
    ("M", 2, 2), # Max pooling, kernel_size, stride
    (3, 192, 1, 1),
    ("M", 2, 2),
    (1, 128, 1, 0)
]

OUT_CNN = [
    # Tuple: (kernel_size, n_filters, stride, padding)
    (7, 64, 2, 3),
    ("M", 2, 2), # Max pooling, kernel_size, stride
    (3, 192, 1, 1),
    ("M", 2, 2),
    (1, 128, 1, 0)
]

DEVICE = "cpu"

m = TimeModel(IN_CNN, OUT_CNN).to(DEVICE)
loss_fn = TimeSplitLoss()
opt = Adam(m.parameters(), lr=1e-3, weight_decay=0)

test_dataset = TDADataset(TEST_DB)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    num_workers=1,
    pin_memory=False,
    shuffle=False,
    drop_last=False
)

def train_fn():
    loop = tqdm(test_loader, leave=True)

    for a, b, y in loop:            
        a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
        out = m(a, b)
        loss = loss_fn(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())


if __name__ == '__main__':
     for epoch in range(100):
        train_fn()