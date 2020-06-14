from __future__ import print_function, division
import os
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from tools import Rescale, RandomCrop, Normalize, ToTensor, DataSplit
import torch.nn.functional as F
from tqdm import tqdm

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated



class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Create dataset, transform and split
dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                root_dir='data/faces/',
                                transform=transforms.Compose([Rescale(256),RandomCrop(224),Normalize(),ToTensor()]))
# split = DataSplit(dataset, shuffle=True, val_train_split=0)

# data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
# train_loader, val_loader, test_loader = split.get_split(batch_size=1, num_workers=4)


X = torch.Tensor([(i['image']).numpy() for i in dataset]).view(-1,224,224)
y = torch.Tensor([(i['landmarks']).numpy() for i in dataset])


val_size = int(len(X)*0.2)
train_X = X[:-val_size]
train_y = y[:-val_size]
test_X = X[-val_size:]
test_y = y[-val_size:]


class Model(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        self.conv1 = nn.Conv2d(1, 32, 5) # input is 1 image, 32 output channels, 5x5 kernel / window
        self.conv2 = nn.Conv2d(32, 64, 5) # input is 32, bc the first layer output 32. Then we say the output will be 64 channels, 5x5 kernel / window
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(224,224).view(-1,1,224,224)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 136) # 512 in, 136 out bc we're doing 68x2 facial landmarks

    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # bc this is our output layer. No activation here.
        return F.softmax(x, dim=1)



model = Model()
optimizer = optim.Adam(model.parameters(), lr = 0.1)
loss_function = nn.SmoothL1Loss()
EPOCHS = 3
BATCH_SIZE = 5

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):

        batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 224, 224)
        batch_y = train_y[i:i+BATCH_SIZE].view(-1, 68*2)


        model.zero_grad()

        outputs = model(batch_X)

        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update

    print('Epoch: {}. Loss: {}'.format(epoch+1, loss))







# fig = plt.figure()
# for i in range(len(dataset)):
#     sample = dataset[i]
#
#     print(i, sample['image'].shape, sample['landmarks'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_landmarks(**sample)
#
#     if i == 3:
#         plt.show()
#         break
