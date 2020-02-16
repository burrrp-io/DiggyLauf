import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
from PIL import Image

cats_path = "/app/data/root/train/"


def loader(liste):
    dataset = []
    for img, cute in liste:
        img_path = cats_path + img

        pic = Image.open(img_path)
        transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32,32)),
                    torchvision.transforms.ToTensor()
                ])
        scaled = transform(pic)
        dataset.append((scaled, cute))
        
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def train(num_epochs, loader, model):
    #Define the optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(loader):
            # Move images and labels to gpu if available
#             if cuda_avail:
#                 images = Variable(images.cuda())
#                 labels = Variable(labels.cuda())

            # Clear all accumulated gradients
            optimizer.zero_grad()
            # Predict classes using images from the test set
            outputs = model(images)
            # Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs, labels)
            # Backpropagate the loss
            loss.backward()

            # Adjust parameters according to the computed gradients
            optimizer.step()
            
            train_loss += loss.item() * images.size(0) #loss.cpu().data[0] * images.size(0)
            _, prediction = torch.max(outputs.data, 1)
            
            train_acc += torch.sum(prediction == labels.data)
            if i > 20:
                break

        # Call the learning rate adjustment function
#         adjust_learning_rate(epoch)

        # Compute the average acc and loss over all 50000 training images
        train_acc = train_acc.float() / ((i+1)*64)#/ 37500
        train_loss = train_loss / ((i+1)*64)

        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {}".format(epoch, train_acc, train_loss))
        model.eval()
    return model

def learn(liste, model):
    train_loader = loader(liste)
    model = train(1, train_loader, model)
    return model