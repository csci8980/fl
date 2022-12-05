"""
    CNN model
"""
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import copy


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


def train(num_epochs, cnn, loaders,model_name):
    """
    ML train function
    :param num_epochs:
    :param cnn:
    :param loaders:
    :return:
    """

    # mq.append(logger.get_str(f'Local training starts'))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    deepcopy_cnn = copy.deepcopy(cnn)
    global_weight_collector = list(deepcopy_cnn.parameters())
    tau = 0

    cnn.train()
    # Train the model
    # total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            #FedProx
            if model_name == "FedProx" or "FedMix":
                mu = 0.001
                fed_prox_reg = 0.0
                for param_index, param in enumerate(cnn.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    print(param)
                    print(global_weight_collector[param_index])
                loss += fed_prox_reg

            if model_name == "FedNova" or "FedMix":
                tau = tau + 1

            # clear gradients for this training step
            optimizer.zero_grad()
            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            #if (i + 1) % 100 == 0:
                # mq.append(logger.get_str(f'Local epoch is {epoch + 1}. Loss is {loss.item()}'))
                # print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                #pass
    return tau


def test(model, loaders):
    """
    ML test function
    :param model:
    :param loaders:
    :return:
    """
    model.eval()
    with torch.no_grad():
        # correct = 0
        # total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = model(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass
        return accuracy


def update_model(model, train_data, test_data,model_name):
    """
    load data, train and test model
    :param model:
    :param train_data:
    :param test_data:
    :return:
    """
    loaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
               'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1), }

    num_epochs = 1
    tau = train(num_epochs, model, loaders, model_name)
    print("tau is", tau)
    accuracy = test(model, loaders)
    return model, accuracy, tau
