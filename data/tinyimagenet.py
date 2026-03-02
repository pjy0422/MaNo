import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_tinyimagenet(corruption_type,
                       clean_cifar_path,
                       corruption_cifar_path,
                       corruption_severity=0,
                       datatype='test',
                       skip_resize=False):

    assert datatype == 'test' or datatype == 'train'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if corruption_type == 'clean':
        if skip_resize:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        dataset = datasets.ImageFolder(root=clean_cifar_path + '/' + datatype,
                                       transform=transform)
    elif corruption_type == 'clean_matching':
        if skip_resize:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        dataset = datasets.ImageFolder(root=clean_cifar_path, transform=transform)
    else:
        if skip_resize:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        dataset = datasets.ImageFolder(root=corruption_cifar_path + '/' + corruption_type + '/' + str(corruption_severity),
                                       transform=transform)
    return dataset