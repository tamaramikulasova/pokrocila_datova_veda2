import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ImageFile.LOAD_TRUNCATED_IMAGES = True

#treba transformovať na 28x28 -> nesmies zabudnut
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('L')),  #odstráni RGBA/paletu
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dir = r"C:\Users\Tamara\Desktop\7.semester\PDA\train_data"
test_dir  = r"C:\Users\Tamara\Documents\GitHub\pokrocila_datova_veda2\dataset\test_data"


train_set = datasets.ImageFolder(root=train_dir, transform=transform)
test_set  = datasets.ImageFolder(root=test_dir,  transform=transform)

num_classes = len(train_set.classes)
print("Triedy:", train_set.classes)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=0, pin_memory=True)


x0, y0 = next(iter(train_loader))
print("Batch shape:", x0.shape)

#model
class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.flatten(x)          # [B, 784]
        x = F.relu(self.fc1(x))
        return self.fc2(x)           # logits

model = DenseNet(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#trening
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/5] Loss: {running_loss/len(train_loader):.4f}")

#uspešnosť
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")