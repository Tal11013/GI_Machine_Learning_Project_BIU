import torch
import wandb
from torchvision.models import InceptionOutputs
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# this is the train function
def train(model, train_loader, criterion, optimizer, number_of_epochs, batch_size):
    # set the model into train mode
    model.train()

    for epoch in range(number_of_epochs):
        train_loss = 0
        correct = 0
        total = 0
        all_probs, all_targets = [], []

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

            # if y_pred type is InceptionOutputs, then take the logits
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

            with torch.no_grad():
                if y_pred.shape[1] == 1:  # single-logit binary
                    probs_pos = torch.sigmoid(y_pred.squeeze())
                    probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
                else:  # 2-class or multi-class
                    probs = torch.softmax(y_pred, dim=1)
            all_probs.append(probs.cpu())
            all_targets.append(y_label.cpu())

        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct / total

        all_probs = torch.cat(all_probs)
        all_targets = torch.cat(all_targets)
        preds = all_probs.argmax(1)

        precision = precision_score(all_targets, preds, average="macro", zero_division=0)
        recall = recall_score(all_targets, preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, preds, average="macro", zero_division=0)

        try:
            if all_probs.shape[1] == 2:  # binary (two columns)
                auroc = roc_auc_score(all_targets, all_probs[:, 1])
                auprc = average_precision_score(all_targets, all_probs[:, 1])
            else:  # multi-class
                auroc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
                auprc = average_precision_score(all_targets, all_probs, average="macro")
        except ValueError:  # happens if only one class present
            auroc = float("nan")
            auprc = float("nan")

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Train Precision": precision,
            "Train Recall": recall,
            "Train F1": f1,
            "Train AUROC": auroc,
            "Train AUPRC": auprc,
        })

        print('Epoch: {} \tTraining Loss: {:.6f} \tAccuracy: {:.2f}%'.format(
              epoch + 1, train_loss, 100 * train_accuracy))
        print(f'P:{precision:.3f}  R:{recall:.3f}  F1:{f1:.3f}  AUROC:{auroc:.3f}  AUPRC:{auprc:.3f}')
    return model


# this is the test function
def test(model, test_loader, criterion, batch_size):
    test_loss = 0
    # set the model into evaluation mode
    model.eval()
    correct = 0
    all_probs, all_targets = [], []

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

        with torch.no_grad():
            if y_pred.shape[1] == 1:
                probs_pos = torch.sigmoid(y_pred.squeeze())
                probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
            else:
                probs = torch.softmax(y_pred, dim=1)
        all_probs.append(probs.cpu())
        all_targets.append(y_label.cpu())

        # count the number of correct predictions
        y_pred = y_pred.max(1, keepdim=True)[1]
        for index in range(batch_size):
            if y_label[index] == y_pred[index]:
                correct += 1

    # print the test loss and accuracy
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    preds = all_probs.argmax(1)

    precision = precision_score(all_targets, preds, average="macro", zero_division=0)
    recall = recall_score(all_targets, preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, preds, average="macro", zero_division=0)

    try:
        if all_probs.shape[1] == 2:
            auroc = roc_auc_score(all_targets, all_probs[:, 1])
            auprc = average_precision_score(all_targets, all_probs[:, 1])
        else:
            auroc = roc_auc_score(all_targets, all_probs, multi_class="ovr", average="macro")
            auprc = average_precision_score(all_targets, all_probs, average="macro")
    except ValueError:
        auroc = float("nan")
        auprc = float("nan")

    wandb.log({
        "Test loss": test_loss,
        "Test accuracy": accuracy,
        "Test Precision": precision,
        "Test Recall": recall,
        "Test F1": f1,
        "Test AUROC": auroc,
        "Test AUPRC": auprc,
    })

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    print(f'P:{precision:.3f}  R:{recall:.3f}  F1:{f1:.3f}  AUROC:{auroc:.3f}  AUPRC:{auprc:.3f}\n')

    return accuracy, test_loss
