import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class CNN_DPD_Micro(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=6, kernel_size=3, padding=1)
        self.act1  = nn.Tanh()

        self.conv2 = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=1, padding=0)
        self.act2  = nn.Tanh()

        self.conv3 = nn.Conv1d(in_channels=6, out_channels=2, kernel_size=3, padding=1)

    def forward(self, x, return_delta=False):
        x_t = x.transpose(1, 2)
        feat = self.act1(self.conv1(x_t))
        feat = self.act2(self.conv2(feat))
        delta = self.conv3(feat)
        out = x_t + delta
        if return_delta:
            return out.transpose(1, 2), delta
        return out.transpose(1, 2)


def plot_signals_200(objects, targets, preds=None, title="Signal Comparison", num_points=200):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(objects[:num_points, 0], marker="o", linestyle='-', label="Input (Distorted) I", alpha=0.7)
    plt.plot(targets[:num_points, 0], marker="x", linestyle='-', label="Target (Ideal) I",    alpha=0.7)
    if preds is not None:
        plt.plot(preds[:num_points, 0], marker="s", linestyle='-', label="Predicted I", alpha=0.7)
    plt.grid(); plt.legend()
    plt.title(f"{title} - Real Part (I)")

    plt.subplot(2, 1, 2)
    plt.plot(objects[:num_points, 1], marker="o", linestyle='-', label="Input (Distorted) Q", alpha=0.7)
    plt.plot(targets[:num_points, 1], marker="x", linestyle='-', label="Target (Ideal) Q",    alpha=0.7)
    if preds is not None:
        plt.plot(preds[:num_points, 1], marker="s", linestyle='-', label="Predicted Q", alpha=0.7)
    plt.grid(); plt.legend()
    plt.title(f"{title} - Imaginary Part (Q)")

    plt.tight_layout()
    plt.show()


def data_prepare_cnn(objects, targets, seq_len=128, batch_size=64):
    print(f"Original shapes: objects = {objects.shape}; targets = {targets.shape}")

    num_blocks  = len(objects) // seq_len
    objects_seq = objects[:num_blocks * seq_len].reshape(num_blocks, seq_len, 2)
    targets_seq = targets[:num_blocks * seq_len].reshape(num_blocks, seq_len, 2)

    x_train, x_test, y_train, y_test = train_test_split(
        objects_seq, targets_seq, test_size=0.1, random_state=42
    )

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float().to(DEVICE),
        torch.from_numpy(y_train).float().to(DEVICE),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float().to(DEVICE),
        torch.from_numpy(y_test).float().to(DEVICE),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset,  batch_size=len(test_dataset), shuffle=False)

    return train_dataloader, test_dataloader


def train_cnn(model, train_dataloader, test_dataloader, criterion_train, criterion_test,
              num_epochs, optimizer, scheduler, device):
    train_loss_avg_arr = []
    test_loss_avg_arr  = []

    for epoch in tqdm(range(num_epochs), desc="Training CNN"):
        model.train()
        total_loss_train = 0

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
        total_loss_test = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                pred = model(x)
                loss = criterion_test(pred, y)
                total_loss_test += loss.item()

        avg_test_loss = total_loss_test / len(test_dataloader)
        test_loss_avg_arr.append(avg_test_loss)
        scheduler.step(avg_test_loss)

    return model, train_loss_avg_arr, test_loss_avg_arr


def test_cnn(model, test_dataloader, criterion_test):
    model.eval()
    total_loss_test = 0
    preds   = []
    targets = []
    objects = []

    with torch.no_grad():
        for x, y in test_dataloader:
            pred = model(x)
            loss = criterion_test(pred, y)
            total_loss_test += loss.item()
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            objects.append(x.cpu().numpy())

    avg_test_loss = total_loss_test / len(test_dataloader)
    preds_flat = np.concatenate(preds,   axis=0).reshape(-1, 2)
    targets_flat = np.concatenate(targets, axis=0).reshape(-1, 2)
    objects_flat = np.concatenate(objects, axis=0).reshape(-1, 2)

    return avg_test_loss, preds_flat, targets_flat, objects_flat


def inference_cnn(signal, batch_size, model, device, weights_file, seq_len=128):
    state_dict = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    data = np.column_stack((signal.real, signal.imag))

    # Overlap-add: receptive field = (k1-1)/2 + (k3-1)/2 = 1+1 = 2
    overlap = 4
    step = seq_len - 2 * overlap

    total_needed = overlap + ((len(data) - overlap + step - 1) // step) * step + overlap
    pad_len = max(0, total_needed - len(data))
    if pad_len > 0:
        data = np.pad(data, ((0, pad_len), (0, 0)), mode='reflect')

    output = np.zeros((len(data), 2), dtype=np.float32)
    num_steps = (len(data) - 2 * overlap) // step

    blocks = []
    positions = []
    for i in range(num_steps):
        start = i * step
        end   = start + seq_len
        blocks.append(data[start:end])
        positions.append((start, end))

    blocks_array = np.array(blocks, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(blocks_array), batch_size):
            data_tensor = torch.from_numpy(blocks_array[i:i + batch_size]).float().to(device)
            pred = model(data_tensor).cpu().numpy()
            for j, (start, end) in enumerate(positions[i:i + batch_size]):
                output[start + overlap: end - overlap] = pred[j, overlap: seq_len - overlap]

    output = output[:len(signal)]

    uncovered = np.where(np.all(output == 0, axis=1))[0]
    if len(uncovered) > 0:
        output[uncovered] = data[:len(signal)][uncovered]

    return output[:, 0] + 1j * output[:, 1]


def main(train_en=1, lsb=2, mod_order=64):
    batch_size = 64
    seq_len = 128

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
            weights_file = "qam_64_cnn_2_LSB_weights.pt"
        else:
            objects = np.load(obj_file_64_qam_4lsb)[1000:1000 + 201000]
            targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
            weights_file = "qam_64_cnn_4_LSB_weights.pt"
    else:
        if lsb == 2:
            objects = np.load(obj_file_32_qam_2lsb)[1000:1000 + 201000]
            targets = np.load(trg_file_32_qam)[1000:1000 + 201000]
            weights_file = "qam_32_cnn_2_LSB_weights.pt"
        else:
            objects = np.load(obj_file_32_qam_4lsb)[1000:1000 + 201000]
            targets = np.load(trg_file_32_qam)[1000:1000 + 201000]
            weights_file = "qam_32_cnn_4_LSB_weights.pt"

    objects = np.column_stack((objects.real, objects.imag))
    targets = np.column_stack((targets.real, targets.imag))

    print("First 200 points BEFORE the model learning...")
    plot_signals_200(objects, targets, title = "Raw Data (Before Training)")

    initial_criterion_check = nn.L1Loss()
    MAE_start = initial_criterion_check(
        torch.from_numpy(objects[:20100]).float(),
        torch.from_numpy(targets[:20100]).float()
    )
    print(f'Initial MAE: {MAE_start}')

    train_dataloader, test_dataloader = data_prepare_cnn(
        objects, targets, seq_len=seq_len, batch_size=batch_size
    )

    model = CNN_DPD_Micro().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters in CNN: {total_params}')

    criterion = nn.MSELoss()
    lr = 3e-3
    num_epochs = 1000
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    if train_en:
        model, train_loss_avg_arr, test_loss_avg_arr = train_cnn(
            model, train_dataloader, test_dataloader,
            criterion, criterion, num_epochs, optimizer, scheduler, DEVICE
        )
        torch.save(model.state_dict(), weights_file)

        plt.figure(figsize=(8, 5))
        plt.plot(train_loss_avg_arr, label="Train loss")
        plt.plot(test_loss_avg_arr,  label="Test loss")
        plt.yscale('log')
        plt.grid(); plt.legend()
        plt.title(f"Train and Test Loss (CNN) | {mod_order}-QAM | INL {lsb} LSB")
        plt.show()

    weights = torch.load(weights_file, map_location=DEVICE, weights_only=True)
    model.load_state_dict(weights)
    criterion_test = nn.L1Loss()

    avg_test_loss, preds_flat, targets_flat, objects_flat = test_cnn(
        model, test_dataloader, criterion_test
    )
    print(f"MAE on test set: {avg_test_loss:.6f}")

    plot_signals_200(objects_flat, targets_flat, preds=preds_flat, title="Test Set Results (After Training)")


if __name__ == "__main__":
    MOD_ORDER = 64
    LSB = 2
    TRAIN_EN = 1
    main(train_en=TRAIN_EN, lsb=LSB, mod_order=MOD_ORDER)
    # 122 Parameters