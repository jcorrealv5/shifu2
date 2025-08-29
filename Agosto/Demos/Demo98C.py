import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Dataset Loading and Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dsTrain = datasets.CelebA(root="C:/Users/jhonf/Documents/Shifu/datasets",download=True,
target_type="landmarks",transform=transform)
dataloader = torch.utils.data.DataLoader(dsTrain, batch_size=32, shuffle=True)
totalMuestras = len(dsTrain)
print("Total de Muestras Entren: ", totalMuestras)


# 2. Model Architecture (Example: A simple CNN)
class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(in_features=7*7*1024, out_features=10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkNet()
model = model.to(device)

# 3. Loss Function and Optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    c = 0
    for images, landmarks in dataloader:
        c = c + 1
        images, landmarks = images.to(device), landmarks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, landmarks.float())
        print(f'Epoch [{epoch+1}/{num_epochs}], Item: {c} Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()
    valorPerdida = loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {valorPerdida:.4f}')
    torch.save(model.state_dict(), "CelebA_Custom_Landmarks_" + str(valorPerdida) + ".pt")