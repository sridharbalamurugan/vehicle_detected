
import torchvision.transforms as T
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # 50% flip chance
    return T.Compose(transforms)
