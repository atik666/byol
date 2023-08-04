import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

path = "/home/atik/Documents/BYOL_FSL/data/"

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),])

TRAIN_DATASET = torchvision.datasets.ImageFolder(root=path+'train_10/', transform=TRANSFORM_IMG)
TEST_DATASET = torchvision.datasets.ImageFolder(root=path+'val/', transform=TRANSFORM_IMG)

train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=10,
    shuffle=True,
    drop_last=True,
)

val_loader = DataLoader(
    TEST_DATASET,
    batch_size=10,
    shuffle=True,
    drop_last=True,
)

from torch.utils.data import DataLoader
from torchvision.models import resnet18

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 100)

model.cuda()

model.load_state_dict(torch.load('./models/byol_unsup1.pt'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

optimizer.load_state_dict(torch.load('./models/optimizer_unsup1.pt'))

num_epochs = 25
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0    
    for inputs, labels in train_loader:

        # Forward pass
        outputs = model(inputs.cuda())
    
        # Compute the loss
        loss = criterion(outputs, labels.cuda())
    
        # Zero gradients, backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Compute average training loss for the epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to the GPU if available
            images, labels = images.cuda(), labels.cuda()

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    # Compute validation accuracy for the epoch
    val_accuracy = 100 * total_correct / total_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.2f}%")
    
# save your improved network
torch.save(model.state_dict(), './models/byol_sup.pt')
# Save the optimizer state_dict to a file
torch.save(optimizer.state_dict(), './models/optimizer_sup.pt')

