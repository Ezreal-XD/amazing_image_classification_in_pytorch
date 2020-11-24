import os
import pickle
from torch.utils import data
import torchvision.transforms as transforms
from data.cifar10 import Cifar10DataSet, Cifar10ValDataSet, Cifar10TrainInform, Cifar10TestDataSet


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale, random_mirror, num_workers):
    data_dir = os.path.join('./data/', 'CIFAR10')
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
        # normMean = [0.7314972, 0.55048054, 0.71203065]
        # normStd = [0.23009692, 0.29576236, 0.20847179]
        normTransform = transforms.Normalize(normMean, normStd)
        trainTransform = transforms.Compose([
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

