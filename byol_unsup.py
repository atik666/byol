import torch
from byol_pytorch import BYOL
from torchvision import models
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch import nn

path = "/home/atik/Documents/BYOL_FSL/data/"

TRANSFORM_IMG = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),])

TRAIN_UNLABELED_DATASET = torchvision.datasets.ImageFolder(root=path+'train_unsup/', transform=TRANSFORM_IMG)

unlbl_loader = DataLoader(
    TRAIN_UNLABELED_DATASET,
    batch_size=128,
    shuffle=True,
    drop_last=True,
)

resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(512, 100)
resnet.cuda()
resnet.train()

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
opt1 = torch.optim.Adam(resnet.parameters(), lr=3e-4)

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0  
    for images, _ in tqdm(unlbl_loader):
        loss = learner(images.cuda())
        opt.zero_grad()
        opt1.zero_grad()
        loss.backward()
        opt.step()
        opt1.step()
        learner.update_moving_average() # update moving average of target encoder
        total_loss += loss.item()
    # Compute average training loss for the epoch
    avg_loss = total_loss / len(unlbl_loader)
    print(f"\n Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}")
    
# save your improved network
torch.save(resnet.state_dict(), './models/byol_unsup1.pt')
# Save the optimizer state_dict to a file
torch.save(opt1.state_dict(), './models/optimizer_unsup1.pt')
