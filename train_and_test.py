import torch
from torchvision.models import InceptionOutputs
from tqdm import tqdm


# this is the train function
def train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size):
    # set the model into train mode
    model.train()

    for epoch in range(number_of_epochs):
        train_loss = 0
        correct = 0
        total = 0
        for x_data, y_label in tqdm(train_loader):
            # if cuda is available, move the data to cuda
            if torch.cuda.is_available():
                x_data = x_data.cuda()
                y_label = y_label.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # set requires_grad to True for input tensor
            x_data.requires_grad = True
            # forward step
            y_pred = model(x_data)
            # if y_pres type is InceptionOutputs, then take the logits
            if type(y_pred) == InceptionOutputs:
                y_pred = y_pred.logits
            # calculate the loss
            loss = criterion(y_pred, y_label)
            # backward pass: compute gradient of the loss
            loss.backward()
            # update the weights
            optimizer.step()
            # update running training loss
            train_loss += loss.item()
            # calculate the number of correct predictions in the batch
            predicted = y_pred.max(1, keepdim=True)[1]
            total += y_label.size(0)
            correct += predicted.eq(y_label.view_as(predicted)).sum().item()

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        accuracy = correct / total
        print('Epoch: {} \tTraining Loss: {:.6f} \tAccuracy: {:.6f}'.format(epoch + 1, train_loss, 100 * accuracy))
    return model


# this is the test function
def test(model, test_loader, criterion, batch_size):
    test_loss = 0
    # set the model into evaluation mode
    model.eval()
    correct = 0

    for x_data, y_label in test_loader:
        # if cuda is available, move the data to cuda
        if torch.cuda.is_available():
            x_data = x_data.cuda()
            y_label = y_label.cuda()
        # forward step
        y_pred = model(x_data)
        # calculate the loss
        loss = criterion(y_pred, y_label)
        # update test loss
        test_loss += loss.item()
        # count the number of correct predictions
        y_pred = y_pred.max(1, keepdim=True)[1]
        for index in range(batch_size):
            if y_label[index] == y_pred[index]:
                correct += 1

    # print the test loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
