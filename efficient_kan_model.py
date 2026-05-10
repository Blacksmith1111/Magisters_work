import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from efficient_kan import KAN

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = 1
INPUT_DIM = 1 * WINDOW_SIZE

# Возвращаем твою сверхкомпактную архитектуру! Минимум параметров.
KAN_model = KAN(layers_hidden=[INPUT_DIM, 4, 1],
                grid_size=5,
                spline_order=3, 
                grid_range=[-1.2, 1.2])

def create_windows(data, window_size):
    dim = data.shape[1]
    pad_size = window_size // 2
    padded_data = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
    windows = np.zeros((len(data), window_size * dim))
    for i in range(window_size):
        windows[:, i * dim:(i + 1) * dim] = padded_data[i : i + len(data)]
    return windows

def data_prepare(objects, targets, window_size, batch_size=64):
    objects_win = create_windows(objects, window_size)
    x_train, x_test, y_train, y_test = train_test_split(
        objects_win, targets, test_size = 0.1, random_state = 42
    )
    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float().to(DEVICE),
        torch.from_numpy(y_train).float().to(DEVICE),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test).float().to(DEVICE),
        torch.from_numpy(y_test).float().to(DEVICE),
    )
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader

def train(model, train_dataloader, test_dataloader, criterion_train, criterion_test, num_epochs, optimizer, scheduler, device):
    train_loss_avg_arr = []
    test_loss_avg_arr = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss_train = 0
        total_loss_test = 0
        
        for x, y in train_dataloader:
            optimizer.zero_grad()
            
            # --- DITHERING ---
            # Добавляем шум (примерно пол-шага квантования). 
            # Это заставит 5 сплайнов выучить ГЛАДКУЮ кривую INL.
            noise = torch.randn_like(x) * 0.03
            
            # Ограничиваем, чтобы зашумленный вход не вылетел за пределы сетки KAN
            x_noisy = torch.clamp(x + noise, -1.19, 1.19)
            
            # В сплайны летит зашумленный сигнал, а прямой путь Residual остается чистым
            pred = model(x_noisy)
            # -----------------
            
            loss = criterion_train(pred, y)
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss_train / len(train_dataloader)
        train_loss_avg_arr.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                pred = model(x) # На тесте шум уже не нужен
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
            pred = model(x)
            loss = criterion_test(pred, y)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            total_loss_test += loss.item()
    avg_test_loss = total_loss_test / len(test_data)
    return avg_test_loss, np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)

def inference_kan(signal, batch_size, model, device, weights_file, window_size=WINDOW_SIZE, max_val=1.0):
    state_dict = torch.load(weights_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 1. Строгая нормализация
    data_real_norm = signal.real.reshape(-1, 1) / max_val
    data_imag_norm = signal.imag.reshape(-1, 1) / max_val
    
    # 2. ЖЕСТКАЯ ЗАЩИТА: отрезаем редкие пики RRC-фильтра, чтобы сплайны не взорвались
    data_real_clipped = np.clip(data_real_norm, -1.19, 1.19)
    data_imag_clipped = np.clip(data_imag_norm, -1.19, 1.19)
    
    win_real = create_windows(data_real_clipped, window_size)
    win_imag = create_windows(data_imag_clipped, window_size)
    
    output_real = np.zeros((len(win_real), 1))
    output_imag = np.zeros((len(win_imag), 1))
    
    with torch.no_grad():
        for i in range(0, len(win_real), batch_size):
            t_real = torch.from_numpy(win_real[i:i + batch_size]).float().to(device)
            output_real[i:i + batch_size] = model(t_real).cpu().numpy()
            
        for i in range(0, len(win_imag), batch_size):
            t_imag = torch.from_numpy(win_imag[i:i + batch_size]).float().to(device)
            output_imag[i:i + batch_size] = model(t_imag).cpu().numpy()
            
    # 3. Возвращаем дельту в исходном масштабе канала
    correction_2d = np.column_stack((output_real[:, 0], output_imag[:, 0])) * max_val
    return correction_2d

def main(train_en = 0, lsb = 2, mod_order = 64):
    batch_size = 8192
    
    try:
        trg_file_64_qam = 'model_targets_64_qam.npy'
        obj_file_64_qam_2_lsb = 'model_objects_64_qam_INL_2_LSB.npy'
        obj_file_64_qam_4_lsb = 'model_objects_64_qam_INL_4_LSB.npy'

        trg_file_32_qam = 'model_targets_32_qam.npy'
        obj_file_32_qam_2_lsb = 'model_objects_32_qam_INL_2_LSB.npy'
        obj_file_32_qam_4_lsb = 'model_objects_32_qam_INL_4_LSB.npy'

        if mod_order == 64:
            if lsb == 2:
                objects = np.load(obj_file_64_qam_2_lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
                weights_file = "qam_64_kan_2_LSB_weights.pt"
            elif lsb == 4:
                objects = np.load(obj_file_64_qam_4_lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
                weights_file = "qam_64_kan_4_LSB_weights.pt"
        else:
            if lsb == 2:
                objects = np.load(obj_file_32_qam_2_lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_32_qam)[1000:1000 + 201000]
                weights_file = "qam_32_kan_2_LSB_weights.pt"
            elif lsb == 4:
                objects = np.load(obj_file_32_qam_4_lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_32_qam)[1000:1000 + 201000]
                weights_file = "qam_32_kan_4_LSB_weights.pt"

    except FileNotFoundError:
        print('Data files not found')

    # Вытягиваем в 1D
    objects = np.concatenate((objects.real, objects.imag)).reshape(-1, 1)
    targets = np.concatenate((targets.real, targets.imag)).reshape(-1, 1)

    # Строгая нормализация
    max_val = np.max(np.abs(objects))
    print(f"STRICT Normalization factor (max_val): {max_val:.4f}")
    
    objects_norm = objects / max_val
    targets_norm = targets / max_val

    train_dataloader, test_dataloader = data_prepare(objects_norm, targets_norm, window_size=WINDOW_SIZE, batch_size=batch_size)

    model = KAN_model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {total_params} !!!') # Убедись, что их мало!
    
    criterion = nn.MSELoss() 
    lr = 5e-3
    num_epochs = 30
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    if train_en:
        model, train_loss_avg_arr, test_loss_avg_arr = train(
            model, train_dataloader, test_dataloader, criterion, criterion, 
            num_epochs, optimizer, scheduler, DEVICE
        )
        torch.save(model.state_dict(), weights_file)
        
        plt.figure(0)
        plt.plot(train_loss_avg_arr, label="Train loss (MSE)")
        plt.plot(test_loss_avg_arr, label="Test loss (MSE)")
        plt.grid()
        plt.legend()
        plt.title("Train and test loss")
        plt.show()
        
    else:
        try:
            weights = torch.load(weights_file, map_location=DEVICE, weights_only=True)
            model.load_state_dict(weights)
            
            avg_test_loss, preds_norm, targets_norm_out = test(model, test_dataloader, criterion)
            
            preds = preds_norm * max_val
            targets_real = targets_norm_out * max_val
            
            final_mse = np.mean((preds - targets_real)**2)
            print(f"Final MSE on test (Real scale) is {final_mse:.6f}")
            print(f'final MAE is {np.mean(np.abs(preds - targets_real))}')
            
            plt.figure(10)
            plt.plot(preds[:200, 0], label="Predicted I", marker='o', markersize=3)
            plt.plot(targets_real[:200, 0], label="Target I", alpha=0.7, marker='.', markersize=3)
            plt.grid()
            plt.legend()
            plt.title("Prediction vs Target (I channel)")
            plt.show()
            
        except FileNotFoundError:
            print(f"Weights file {weights_file} not found")

if __name__ == "__main__":
    TRAIN_EN = 0
    MOD_ORDER = 32
    LSB = 2
    main(train_en = TRAIN_EN, lsb = LSB, mod_order = MOD_ORDER)