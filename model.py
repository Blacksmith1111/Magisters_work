import numpy as np
import torch
import torch.nn as nn
from channel_funcs import quantizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import commpy.modulation as mod
import channel_funcs as cf
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


"""class MLP_model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64 * in_features)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(64 * in_features, 128 * in_features)
        self.fc3 = nn.Linear(128 * in_features, 64 * in_features)
        self.fc4 = nn.Linear(64 * in_features, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x"""

MLP_model = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 2),
)


def data_prepare(objects, targets, batch_size=64):
    print(f"objects.shape = {objects.shape}; targets.shape = {targets.shape}")
    # plt.ion()
    plt.figure(1)
    plt.plot(objects[:1000, 0], marker="o", label="Signal after the ADC")
    plt.plot(
        targets[:1000, 0], marker="o", label="Signal after the initial pulse shaping"
    )
    plt.grid()
    plt.legend()
    plt.figure(2)
    plt.plot(objects[:1000, 1], marker="o", label="Signal after the ADC")
    plt.plot(
        targets[:1000, 1], marker="o", label="Signal after the initial pulse shaping"
    )
    plt.grid()
    plt.legend()
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(
        objects, targets, test_size=0.1, random_state=42
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float().to(DEVICE),
        torch.from_numpy(y_train).float().to(DEVICE),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float().to(DEVICE),
        torch.from_numpy(y_test).float().to(DEVICE),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def train(
    model,
    train_dataloader,
    test_dataloader,
    criterion_train,
    criterion_test,
    num_epochs,
    optimizer,
    scheduler,
    device,
):
    train_loss_avg_arr = []
    test_loss_avg_arr = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss_train = 0
        total_loss_test = 0
        for x, y in train_dataloader:
            optimizer.zero_grad()
            # x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion_train(pred, y)
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss_train / len(train_dataloader)
        train_loss_avg_arr.append(avg_train_loss)
        model.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                # x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion_test(pred, y)
                total_loss_test += loss.item()
        avg_test_loss = total_loss_test / len(test_dataloader)
        test_loss_avg_arr.append(avg_test_loss)
        scheduler.step(avg_test_loss)
    return model, train_loss_avg_arr, test_loss_avg_arr


def test(model, test_data, criterion_test):
    model.eval()
    total_loss_test = 0
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in test_data:
            # x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion_test(pred, y)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            total_loss_test += loss.item()
    avg_test_loss = total_loss_test / len(test_data)
    return avg_test_loss, np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def main():
    TRAIN_EN = 0
    batch_size = 8192
    objects = np.load("objects_64_qam.npy")[:200000]
    targets = np.load("targets_64_qam.npy")[:200000]
    train_dataloader, test_dataloader = data_prepare(
        objects,
        targets,
        batch_size=batch_size,
    )

    model = MLP_model.to(DEVICE)
    # model = torch.compile(model)
    criterion = nn.MSELoss()
    lr = 3e-3
    num_epochs = 700
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=30
    )

    if TRAIN_EN:
        model, train_loss_avg_arr, test_loss_avg_arr = train(
            model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            criterion_train=criterion,
            criterion_test=criterion,
            num_epochs=num_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            device=DEVICE,
        )
        weights_file = "qam_64_mlp_weights.pt"
        torch.save(model.state_dict(), weights_file)
        plt.figure(0)
        plt.plot(train_loss_avg_arr, label="Train loss")
        plt.plot(test_loss_avg_arr, label="Tets loss")
        plt.grid()
        plt.legend()
        plt.title("Train and test loss")
        plt.savefig(f"train_and_test_loss_num_epoch_{num_epochs}.png")
        plt.show()
    else:
        weights = torch.load(
            "qam_64_mlp_weights.pt", map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(weights)
        criterion_test = nn.L1Loss()
        avg_test_loss, preds, targets = test(model, test_dataloader, criterion_test)
        print(f"MAE on test is {avg_test_loss}")
        data_prepare(objects=preds, targets=targets)
        temp = 0


if __name__ == "__main__":
    main()
