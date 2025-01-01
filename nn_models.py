import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import random_split


TRAIN_FRAC = 0.9
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X


class MLP_simple(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU()):
        super(MLP_simple, self).__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.output = nn.Linear(num_units, num_classes)
    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.output(X)
        return X


class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=2, nonlin=nn.ReLU()):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, num_classes)
        self.nonlin = nonlin

    def forward(self, X, **kwargs):
        return self.nonlin(self.dense0(X.float()))


class MLPClassifier():
    def __init__(self, input_dim=9, num_classes=2, lr=0.0002, batch_size=128, train_epochs=30, hidden_size=16,
                 dropout_ratio=0.1, device='cuda', model_mode='single'):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        if model_mode == 'double':
            self.model = MLP(self.input_dim, self.num_classes, num_units=hidden_size,
                         dropout_ratio=dropout_ratio)
        elif model_mode == 'single':
            self.model = MLP_simple(self.input_dim, self.num_classes, num_units=hidden_size)
        elif model_mode == 'linear':
            self.model = Linear(self.input_dim, self.num_classes)
        self.device = device

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_logits(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
             # X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        # softmax = nn.Softmax(dim=1)
        # outputs = softmax(outputs)
        return outputs.detach().cpu().numpy()

    # MLP的输出仅仅为logits，并非概率，因此在该函数中加入softmax
    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
            # X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        return outputs.detach().cpu().numpy()

    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
            # X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
            # y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
            # X = torch.tensor(X)
            # y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cuda'):
        self.device = device

        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a tensor, but got: {}".format(type(X)))
            # X = torch.tensor(X)
            # y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()

        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        best_val_loss = 100000
        best_model_state = None
        train_loss = []
        valid_loss = []
        for epoch in range(0, self.train_epochs + 1):
            running_loss = 0.0
            epoch_steps = 0
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
            train_loss.append(running_loss)
            print("[%d] train loss: %.3f" % (epoch + 1, running_loss))

            self.model.eval()
            total = 0
            correct = 0
            running_loss = 0.0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            valid_loss.append(running_loss)
            val_accuracy = correct / total
            print("[%d] valid loss: %.3f" % (epoch + 1, running_loss))
            print(f"Validation Accuracy: {val_accuracy*100}%")
            if running_loss < best_val_loss:
                best_val_loss = running_loss
                best_model_state = self.model.state_dict()  # 保存当前最佳的模型参数
                print("Saved best model so far")

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Loaded the best model based on validation accuracy.")
        torch.save(self.model.state_dict(), 'best_model.pth')
        return train_loss, valid_loss