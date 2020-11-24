import os
import pickle
from torch.utils import data
from .seed import SeedDataset
# from dataset.cityscapes import CityscapesDataSet, CityscapesTrainInform, CityscapesValDataSet, CityscapesTestDataSet
# from dataset.camvid import CamVidDataSet, CamVidValDataSet, CamVidTrainInform, CamVidTestDataSet


def build_dataset_train(batch_size, num_workers):
    # train_list = os.path.join('..', '..', 'data', 'train.txt')
    # val_list = os.path.join('..', '..', 'data', 'val.txt')
    train_list = "Z:/CreatingCodeHere/seed/data/train.txt"
    val_list = "Z:/CreatingCodeHere/seed/data/val.txt"

    trainLoader = data.DataLoader(SeedDataset(train_list),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    valLoader = data.DataLoader(SeedDataset(val_list),
                                batch_size=1,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=True,
                                drop_last=True)

    return trainLoader, valLoader


def build_dataset_test(batch_size, num_workers):
    test_list = os.path.join('..', '..', 'data', 'test.txt')

    testLoader = data.DataLoader(SeedDataset(test_list),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    return testLoader


if __name__ == '__main__':
    print(os.path.join('..', '..', 'data', 'train' + '.txt'))
