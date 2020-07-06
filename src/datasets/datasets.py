from torchvision import transforms
from src.datasets.cifar10 import CIFAR10, CIFAR10TwoViews
from src.datasets.imagenet import ImageNet, ImageNetTwoViews

DATASET = {
    'cifar10': CIFAR10,
    'imagenet': ImageNet
    'cifar10_2views': CIFAR10TwoViews,
    'imagenet_2views': ImageNetTwoViews,
}

def get_datasets(dataset_name):
    train_transforms, test_transforms = \
        load_default_transforms(dataset_name)
    train_dataset = DATASET[dataset_name](
        train=True,
        image_transforms=train_transforms,
    )
    val_dataset = DATASET[dataset_name](
        train=False,
        image_transforms=test_transforms,
    )
    return train_dataset, val_dataset


def load_default_transforms(dataset):
    # resize imagenet to 256
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms
