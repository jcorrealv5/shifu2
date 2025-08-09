from torchvision import datasets, transforms

print("Demo 92: Trabajando con el DataSet CelebA")
dst = datasets.CelebA(root="datasets",download=True,transform=transforms.ToTensor())
print("DataSet: ", dst)