import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR-10 statistics
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10("CIFAR", train=True, download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR10("CIFAR", train=False, download=True, transform=test_transforms)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return trainloader, testloader
