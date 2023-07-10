import paddle
import io
import numpy as np
import sys
from PIL import Image
from data_process import *
from paddle.vision import transforms

print(paddle.__version__)

vocab = load_vocab("./tmp_data/dict.txt")

class MyDataset(paddle.io.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_data = []
        self.images = images

        with io.open(self.data_dir, "r", encoding='utf8') as fin:
            for line in fin:
                cols = line.strip().split("\t\t\t\t\t")
                if len(cols) != 2:
                    sys.stderr.write("[NOTICE] Error Format Line!")
                    continue
                label = []
                label.append(int(cols[1]))
                wids = cols[0].split(",")
                if len(wids) >= 150:
                    wids = np.array(wids[:150]).astype('int64')
                else:
                    wids = np.concatenate([wids, [vocab["<pad>"]] * (150 - len(wids))]).astype('int64')
                label = np.array(label).astype('int64')
                self.all_data.append((wids, label))

    def __getitem__(self, index):
        data, label = self.all_data[index]
        image = self.images[index]
        # print(len(self.images),len(self.all_data))
        # root="./data"
        # path=os.path.join(root,image)
        img = Image.open(image)

        # Resize the image to 512*412
        img = img.resize((224, 224))
        img = transforms.ToTensor()(img)
        return img, data, label

    def __len__(self):
        return len(self.all_data)


batch_size = 32
train_dataset = MyDataset(train_path)
val_dataset = MyDataset(val_path)
test_dataset = MyDataset(test_path)

train_loader = paddle.io.DataLoader(train_dataset, return_list=True,
                                    batch_size=batch_size, drop_last=True)
val_loader = paddle.io.DataLoader(val_dataset, return_list=True,
                                   batch_size=batch_size, drop_last=True)
test_loader = paddle.io.DataLoader(test_dataset, return_list=True,
                                   batch_size=batch_size, drop_last=False)

