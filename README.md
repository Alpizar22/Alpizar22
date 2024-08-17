
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define batch size
batch_size_train = 64
batch_size_test = 1024

# Define image transform
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True, transform=image_transform)
test_dataset = torchvision.datasets.MNIST('dataset/', train=False, download=True, transform=image_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

# Visualize data
_, (example_datas, labels) = next(enumerate(test_loader))
sample = example_datas[0][0]
plt.imshow(sample, cmap='gray', interpolation='none')
print("Label: " + str(labels[0]))

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define training and testing functions
def train(model, device, train_loader, optimizer, epoch, log_interval=10000):
    model.train()
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    counter = 0
    for batch_idx, (data, target) in enumerate(tk0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        counter += 1
        tk0.set_postfix(loss=(loss.item()*data.size(0) / (counter * train_loader.batch_size)))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return accuracy

# Define different configurations
configurations = [
    {'learning_rate': 0.01, 'optimizer': optim.SGD},
    {'learning_rate': 0.001, 'optimizer': optim.SGD},
    {'learning_rate': 0.01, 'optimizer': optim.Adam},
    {'learning_rate': 0.001, 'optimizer': optim.Adam},
]

num_epochs = 50
results = []

for config in configurations:
    learning_rate = config['learning_rate']
    optimizer_class = config['optimizer']
    
    # Initialize model, optimizer
    device = "cpu"
    model = CNN().to(device)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining with learning_rate={learning_rate} and optimizer={optimizer_class.__name__}")
    
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
    
    results.append((learning_rate, optimizer_class.__name__, accuracy))

# Print results
print("\nFinal Results:")
for lr, opt, acc in results:
    print(f"Learning Rate: {lr}, Optimizer: {opt}, Final Accuracy: {acc:.2f}%")
