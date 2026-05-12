import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from efficient_kan import KAN
from itertools import product

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 1
INPUT_DIM   = 2 * WINDOW_SIZE

'''class KAN_DPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = KAN(layers_hidden=[INPUT_DIM, 4, 2],
                       grid_size=10,
                       spline_order=3,
                       grid_range=[-1.2, 1.2])
    
    def forward(self, x):
        return x + self.kan(x)

KAN_model = KAN_DPD()'''

KAN_model = KAN(layers_hidden=[INPUT_DIM, 4, 2],
                grid_size=10,
                spline_order=3,
                grid_range=[-1.2, 1.2])


def create_windows(data, window_size):
    dim = data.shape[1]
    pad_size = window_size // 2
    padded_data = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
    windows = np.zeros((len(data), window_size * dim))
    for i in range(window_size):
        windows[:, i * dim:(i + 1) * dim] = padded_data[i: i + len(data)]
    return windows


def data_prepare(objects, targets, window_size, batch_size=64):

    objects_win = create_windows(objects, window_size)

    split_idx = int(len(objects_win) * 0.9)
    x_train, x_test = objects_win[:split_idx], objects_win[split_idx:]
    y_train, y_test = targets[:split_idx],     targets[split_idx:]

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float().to(DEVICE),
        torch.from_numpy(y_train).float().to(DEVICE),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float().to(DEVICE),
        torch.from_numpy(y_test).float().to(DEVICE),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader, x_test, y_test


def train(model, train_dataloader, test_dataloader,
          criterion_train, criterion_test, num_epochs,
          optimizer, scheduler, device):
    train_loss_avg_arr = []
    test_loss_avg_arr  = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss_train = 0
        total_loss_test  = 0

        for x, y in train_dataloader:
            optimizer.zero_grad()
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
    preds   = []
    targets = []
    with torch.no_grad():
        for x, y in test_data:
            pred = model(x)
            loss = criterion_test(pred, y)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            total_loss_test += loss.item()
    avg_test_loss = total_loss_test / len(test_data)
    return avg_test_loss, np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def plot_comparison(x_raw, preds, targets, max_val, n=300):

    raw_i = x_raw[:n, 0] * max_val
    raw_q = x_raw[:n, 1] * max_val if x_raw.shape[1] >= 2 else x_raw[:n, 0] * max_val

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = np.arange(n)

    axes[0].plot(t, raw_i, label = "Before the model (input)", alpha=0.7, linestyle='--', color='tab:blue')
    axes[0].plot(t, preds[:n, 0], label="After the model (KAN)", alpha=0.9, color='tab:green')
    axes[0].plot(t, targets[:n, 0], label="Target", alpha=0.7, linestyle=':', color='tab:red')
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("I-channel")
    axes[0].legend(loc='upper right')
    axes[0].grid(True)

    axes[1].plot(t, raw_q, label = "Before the model (input)", alpha=0.7, linestyle='--', color='tab:blue')
    axes[1].plot(t, preds[:n, 1], label = "After the model (KAN)", alpha=0.9, color='tab:green')
    axes[1].plot(t, targets[:n, 1], label = "Target", alpha=0.7, linestyle=':',  color='tab:red')
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Sample")
    axes[1].set_title("Q-channel")
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def print_mae_comparison(x_raw, preds, targets, max_val):

    raw_i = x_raw[:, 0] * max_val
    raw_q = x_raw[:, 1] * max_val if x_raw.shape[1] >= 2 else x_raw[:, 0] * max_val

    tgt_i = targets[:, 0]
    tgt_q = targets[:, 1]

    mae_before_i = np.mean(np.abs(raw_i - tgt_i))
    mae_before_q = np.mean(np.abs(raw_q - tgt_q))
    mae_before   = np.mean(np.abs(
        np.column_stack((raw_i, raw_q)) - targets
    ))

    mae_after_i = np.mean(np.abs(preds[:, 0] - tgt_i))
    mae_after_q = np.mean(np.abs(preds[:, 1] - tgt_q))
    mae_after   = np.mean(np.abs(preds - targets))

    print("=" * 45)
    print(f"{'Metric':<20} {'Before the model':>10} {'After KAN':>12}")
    print("=" * 45)
    print(f"{'MAE I-channel':<20} {mae_before_i:>10.6f} {mae_after_i:>12.6f}")
    print(f"{'MAE Q-channel':<20} {mae_before_q:>10.6f} {mae_after_q:>12.6f}")
    print(f"{'MAE sum':<20} {mae_before:>10.6f} {mae_after:>12.6f}")
    print("=" * 45)
    improvement = (1 - mae_after / mae_before) * 100
    print(f"MAE improvement: {improvement:.1f}%")
    print("=" * 45)


def inference_kan(signal, batch_size, model, device, weights_file,
                  window_size=WINDOW_SIZE, max_val=1.0):
    state_dict = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    data_iq = np.column_stack((signal.real, signal.imag)) / max_val
    data_iq = np.clip(data_iq, -1.19, 1.19)

    win_iq = create_windows(data_iq, window_size)
    output_iq = np.zeros((len(win_iq), 2))

    with torch.no_grad():
        for i in range(0, len(win_iq), batch_size):
            t = torch.from_numpy(win_iq[i:i + batch_size]).float().to(device)
            output_iq[i:i + batch_size] = model(t).cpu().numpy()

    return output_iq * max_val


def main(train_en=0, lsb=2, mod_order=64):
    batch_size = 8192

    try:
        trg_file_64_qam = 'model_targets_64_qam.npy'
        obj_file_64_qam_2lsb = 'model_objects_64_qam_INL_2_LSB.npy'
        obj_file_64_qam_4lsb = 'model_objects_64_qam_INL_4_LSB.npy'
        trg_file_32_qam = 'model_targets_32_qam.npy'
        obj_file_32_qam_2lsb = 'model_objects_32_qam_INL_2_LSB.npy'
        obj_file_32_qam_4lsb = 'model_objects_32_qam_INL_4_LSB.npy'

        if mod_order == 64:
            if lsb == 2:
                objects = np.load(obj_file_64_qam_2lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
                weights_file = "qam_64_kan_2_LSB_weights.pt"
            elif lsb == 4:
                objects = np.load(obj_file_64_qam_4lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
                weights_file = "qam_64_kan_4_LSB_weights.pt"
        else:
            if lsb == 2:
                objects = np.load(obj_file_32_qam_2lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_32_qam)[1000:1000 + 201000]
                weights_file = "qam_32_kan_2_LSB_weights.pt"
            elif lsb == 4:
                objects = np.load(obj_file_32_qam_4lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_32_qam)[1000:1000 + 201000]
                weights_file = "qam_32_kan_4_LSB_weights.pt"

    except FileNotFoundError:
        print('Data files not found')
        return

    objects = np.column_stack((objects.real, objects.imag))
    targets = np.column_stack((targets.real, targets.imag))

    max_val = np.max(np.abs(objects))
    print(f'Max_val for mod order = {mod_order}; inl val = {lsb} == {max_val}')
    objects_norm = objects / max_val
    targets_norm = targets / max_val

    objects_clipped = np.clip(objects_norm, -1.19, 1.19)

    print(f"Normalization max_val: {max_val:.4f}")

    train_dataloader, test_dataloader, x_test_raw, y_test_raw = data_prepare(
        objects_clipped, targets_norm,
        window_size=WINDOW_SIZE, batch_size=batch_size
    )

    model = KAN_model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    criterion = nn.MSELoss()
    lr = 5e-3
    num_epochs = 50
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    if train_en:
        model, train_loss_arr, test_loss_arr = train(
            model, train_dataloader, test_dataloader,
            criterion, criterion, num_epochs, optimizer, scheduler, DEVICE
        )
        torch.save(model.state_dict(), weights_file)

        plt.figure(0)
        plt.plot(train_loss_arr, label="Train MSE")
        plt.plot(test_loss_arr,  label="Test MSE")
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.title("KAN Train/Test Loss")
        #plt.show()

    else:
        try:
            weights = torch.load(weights_file, map_location=DEVICE, weights_only=True)
            model.load_state_dict(weights)

            _, preds_norm, targets_norm_out = test(model, test_dataloader, criterion)

            preds       = preds_norm * max_val
            targets_out = targets_norm_out * max_val

            print_mae_comparison(x_test_raw, preds, targets_out, max_val)
            plot_comparison(x_test_raw, preds, targets_out, max_val, n=300)

        except FileNotFoundError:
            print(f"Weights file {weights_file} not found")


if __name__ == "__main__":
    TRAIN_EN = 1
    MOD_ORDER = 64
    LSB = 4
    lsb_arr = [2, 4]
    mod_order_arr = [32, 64]
    test_cases = list(product(lsb_arr, mod_order_arr))
    # 160 Parameters/ 240
    for lsb, mod in test_cases:
        main(train_en = TRAIN_EN, lsb = lsb, mod_order = mod)
    #main(train_en=TRAIN_EN, lsb=LSB, mod_order=MOD_ORDER)