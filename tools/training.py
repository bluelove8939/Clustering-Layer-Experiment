import torch
from tools.progressbar import progressbar


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train(dataloader, model, loss_fn, optimizer, verbose=2):
    size = len(dataloader.dataset)  # size of the dataset
    model.train()                   # turn the model into train mode
    for batch, (X, y) in enumerate(dataloader):  # each index of dataloader will be batch index
        X, y = X.to(device), y.to(device)        # extract input and output

        # Compute prediction error
        pred = model(X)          # predict model
        # print(pred.shape, y.shape)
        loss = loss_fn(pred, y)  # calculate loss

        # Backpropagation
        optimizer.zero_grad()  # gradient initialization (just because torch accumulates gradient)
        loss.backward()        # backward propagate with the loss value (or vector)
        optimizer.step()       # update parameters

        loss, current = loss.item(), (batch+1) * dataloader.batch_size

        if verbose == 1:
            print(f"\rloss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="")
        elif verbose:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    if verbose == 1: print('\n')
    elif verbose: print()


def test(dataloader, model, loss_fn, verbose=2):
    size = len(dataloader.dataset)  # dataset size
    num_batches = len(dataloader)   # the number of batches
    model.eval()                    # convert model into evaluation mode
    test_loss, correct = 0, 0       # check total loss and count correctness
    with torch.no_grad():           # set all of the gradient into zero
        for didx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)     # extract input and output
            pred = model(X)                       # predict with the given model
            test_loss += loss_fn(pred, y).item()  # acculmulate total loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()  # count correctness

            if verbose == 1:
                print(f"\rtest status: {progressbar(didx+1, len(dataloader), scale=50)} {(didx+1) / len(dataloader) * 100:2.0f}%",end='')
            elif verbose:
                print(f"test status: {progressbar(didx+1, len(dataloader), scale=50)} {(didx+1) / len(dataloader) * 100:2.0f}%")

    test_loss /= num_batches   # make an average of the total loss
    correct /= size            # make an average with correctness count

    if verbose:
        if verbose == 1: print()
        print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

    return 100 * correct, test_loss