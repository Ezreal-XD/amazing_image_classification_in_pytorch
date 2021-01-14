import os
import pickle
from torch.utils import data
import torchvision.transforms as transforms
from data.cifar10 import Cifar10DataSet, Cifar10ValDataSet, Cifar10TrainInform, Cifar10TestDataSet
from data.cifar100 import Cifar100DataSet, Cifar100ValDataSet, Cifar100TrainInform, Cifar100TestDataSet
from data.hansim import HanSimDataSet, HanSimValDataSet, HanSimTrainInform, HanSimTestDataSet


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.join('./data/', dataset)
    dataset_list = dataset + '_trainval.txt'
    train_data_list = os.path.join(data_dir, dataset + '_' + train_type + '.txt')
    val_data_list = os.path.join(data_dir, dataset + '_val' + '.txt')
    inform_data_file = os.path.join('./data/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == 'cifar10':
            dataCollect = Cifar10TrainInform(data_dir, 10, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'cifar100':
            dataCollect = Cifar100TrainInform(data_dir, 100, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'hansim':
            dataCollect = HanSimTrainInform(data_dir, 3755, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)

        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cifar10":
        normMean = datas['mean']
        normStd = datas['std']
        # normMean = [0.4914, 0.4822, 0.4465],
        # normStd = [0.2023, 0.1994, 0.2010]
        normTransform = transforms.Normalize(normMean, normStd)
        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])

        validTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

        trainLoader = data.DataLoader(
            Cifar10DataSet(data_dir, train_data_list, transform=trainTransform),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            Cifar10ValDataSet(data_dir, val_data_list, transform=validTransform),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader

    elif dataset == "cifar100":
        normMean = datas['mean']
        normStd = datas['std']
        # normMean = [0.4914, 0.4822, 0.4465],
        # normStd = [0.2023, 0.1994, 0.2010]
        normTransform = transforms.Normalize(normMean, normStd)
        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])

        validTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

        trainLoader = data.DataLoader(
            Cifar10DataSet(data_dir, train_data_list, transform=trainTransform),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            Cifar10ValDataSet(data_dir, val_data_list, transform=validTransform),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader

    elif dataset == "hansim":
        normMean = datas['mean']
        normStd = datas['std']
        normTransform = transforms.Normalize(normMean, normStd)
        trainTransform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.1, 0.33), ratio=(0.2, 5)),
            # transforms.RandomErasing(p=0.3, scale=(0.05, 0.15), ratio=(0.9, 1.1)),
            normTransform
        ])

        validTransform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            # transforms.RandomErasing(p=0.5, scale=(0.1, 0.33), ratio=(0.2, 5)),
            # transforms.RandomErasing(p=0.3, scale=(0.05, 0.15), ratio=(0.9, 1.1)),
            normTransform
        ])

        trainLoader = data.DataLoader(
            HanSimDataSet(data_dir, train_data_list, transform=trainTransform),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            HanSimValDataSet(data_dir, val_data_list, transform=validTransform),
            batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True)

        return datas, trainLoader, valLoader


def build_dataset_test(dataset, num_workers, none_gt=False):
    data_dir = os.path.join('./data/', 'CIFAR10')
    dataset = 'cls'
    dataset_list = dataset + '_trainval.txt'
    test_data_list = os.path.join(data_dir, dataset + '_test' + '.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    # inform_data_file collect the information of mean, std and weigth_class
    if not os.path.isfile(inform_data_file):
        print("%s is not found" % (inform_data_file))
        if dataset == 'cifar10':
            dataCollect = Cifar10TrainInform(data_dir, 10, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        if dataset == 'cifar100':
            dataCollect = Cifar100TrainInform(data_dir, 100, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        elif dataset == 'hansim':
            dataCollect = HanSimTrainInform(data_dir, 3755, train_set_file=dataset_list,
                                            inform_data_file=inform_data_file)
        else:
            raise NotImplementedError(
                "This repository now supports two datasets: cityscapes and camvid, %s is not included" % dataset)
        
        datas = dataCollect.collectDataAndSave()
        if datas is None:
            print("error while pickling data. Please check.")
            exit(-1)
    else:
        print("find file: ", str(inform_data_file))
        datas = pickle.load(open(inform_data_file, "rb"))

    if dataset == "cifar10":

        normMean = [0.49413836, 0.48498306, 0.45054787]
        normStd = [0.20214662, 0.1993526, 0.20140734]
        normTransform = transforms.Normalize(normMean, normStd)
        testTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

        testLoader = data.DataLoader(
            Cifar10ValDataSet(data_dir, test_data_list, transform=testTransform),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return datas, testLoader

