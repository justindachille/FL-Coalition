import torch
from torch.utils.data import DataLoader, TensorDataset

class Client():
    def __init__(self, id, X_train, Y_train, X_test, Y_test, batch_size):
        self.id = id
        self.batch_size = batch_size
        self.train_data = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        self.test_data = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    def train(self, model, epochs, lr):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()

    def test(self, model):
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        return acc
