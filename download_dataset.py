from torchvision import datasets, transforms

# Define the directory where the dataset will be stored
root_dir = './data/flowers102'

# Define any transformations for the images (e.g., resizing, cropping, etc.)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load the training set of the Flowers102 dataset
train_dataset = datasets.Flowers102(root=root_dir, split='train', transform=transform, download=True)

# Similarly, you can load the validation and test sets by changing the split argument
val_dataset = datasets.Flowers102(root=root_dir, split='val', transform=transform, download=True)
test_dataset = datasets.Flowers102(root=root_dir, split='test', transform=transform, download=True)
