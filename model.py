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


def dataset_gen(signal: np.ndarray, resolution: int):
    out_noisy, _ = quantizer(signal, resolution, INL_en=1)
    out_clean, _ = quantizer(signal, resolution, INL_en=0)
    return out_noisy, out_clean


def complex_to_real(signal):
    return np.stack([signal.real, signal.imag], axis=-1).astype(np.float32)


class MLP_model(nn.Module):
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
        return x


def data_prepare(device, batch_size=64):
    # Parameters
    BITS_NUM = 120_000
    MOD_ORDER = 64
    FS = 10e3
    SPS = 10
    F_SYM = FS / SPS
    TS = 1 / F_SYM
    ROLLOFF = 0.25
    FILTER_SPAN = 10
    DAC_BITS = 6

    bits = np.random.randint(0, 2, BITS_NUM)
    qam = mod.QAMModem(MOD_ORDER)
    symbol_signal = qam.modulate(bits)

    up_signal = cf.upsample(symbol_signal, SPS)

    shaped_signal = cf.pulse_shaping(
        up_signal,
        rolloff=ROLLOFF,
        filter_span=FILTER_SPAN,
        sps=SPS,
        Fs=FS,
        Ts=TS,
        normaliztion="L2",
    )

    cf.spectrum_plot(shaped_signal, FS, title="Original baseband signal spectrum")
    quantized_noisy, quantized_clean = dataset_gen(shaped_signal, resolution=DAC_BITS)
    quantized_noisy = complex_to_real(quantized_noisy)
    quantized_clean = complex_to_real(quantized_clean)
    x_train, x_test, y_train, y_test = train_test_split(
        quantized_noisy, quantized_clean, test_size=0.1, random_state=42
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).to(DEVICE), torch.from_numpy(y_train).to(DEVICE)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).to(DEVICE), torch.from_numpy(y_test).to(DEVICE)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=True,
        # num_workers=6,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        # pin_memory=True,
        # num_workers=6,
    )
    return train_dataloader, test_dataloader, x_train.shape[-1]


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


def main():
    TRAIN_EN = 1
    batch_size = 8192
    train_dataloader, test_dataloader, num_features = data_prepare(
        device=DEVICE, batch_size=batch_size
    )

    model = MLP_model(num_features).to(DEVICE)
    # model = torch.compile(model)
    criterion = nn.MSELoss()
    lr = 3e-3
    num_epochs = 500
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
        plt.figure(0)
        plt.plot(np.arange(num_epochs) + 1, train_loss_avg_arr, label="Train loss")
        plt.plot(np.arange(num_epochs) + 1, test_loss_avg_arr, label="Tets loss")
        plt.grid()
        plt.legend()
        plt.title("Train and test loss")
        plt.savefig(f"train_and_test_loss_num_epoch_{num_epochs}.png")
        plt.show()


if __name__ == "__main__":
    main()
