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
INPUT_DIM = 2 * WINDOW_SIZE

KAN_model = KAN(layers_hidden=[INPUT_DIM, 4, 2],
                grid_size=5,
                spline_order=3, 
                grid_range=[-1.2, 1.2])

def create_windows(data, window_size):
    pad_size = window_size // 2
    padded_data = np.pad(data, ((pad_size, pad_size), (0, 0)), mode='edge')
    
    windows = np.zeros((len(data), window_size * 2))
    for i in range(window_size):
        windows[:, i * 2:(i + 1) * 2] = padded_data[i : i + len(data)]
        
    return windows

def data_prepare(objects, targets, window_size, batch_size=64):
    print(f"Windowing data. Objects shape: {objects.shape}, Range: {objects.min():.3f} to {objects.max():.3f}")
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
    
    data = np.column_stack((signal.real, signal.imag))
    
    data_norm = data / max_val
    data_win = create_windows(data_norm, window_size)
    output_norm = np.zeros_like(data_norm)
    
    with torch.no_grad():
        for i in range(0, len(data_win), batch_size):
            data_tensor = torch.from_numpy(data_win[i:i + batch_size]).float().to(device)
            pred = model(data_tensor)
            output_norm[i:i + batch_size] = pred.cpu().numpy()
            
    output = output_norm * max_val
    return output

def main(train_en = 0, lsb = 2, mod_order = 64):
    batch_size = 8192
    
    try:
        trg_file_64_qam = 'model_targets_64_qam.npy'
        obj_file_64_qam_2_lsb = 'model_objects_64_qam_INL_2_LSB.npy'
        obj_file_64_qam_4_lsb = 'model_objects_64_qam_INL_4_LSB.npy'

        if mod_order == 64:
            if lsb == 2:
                objects = np.load(obj_file_64_qam_2_lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
                weights_file = "qam_64_kan_2_LSB_weights.pt"
            elif lsb == 4:
                objects = np.load(obj_file_64_qam_4_lsb)[1000:1000 + 201000]
                targets = np.load(trg_file_64_qam)[1000:1000 + 201000]
                weights_file = "qam_64_kan_4_LSB_weights.pt"
        objects = np.load("model_objects_64_qam.npy")[1000:1000 + 201000]
        targets = np.load("model_targets_64_qam.npy")[1000:1000 + 201000]
    except FileNotFoundError:
        print('Data files not found')

    objects = np.column_stack((objects.real, objects.imag))
    targets = np.column_stack((targets.real, targets.imag))

    max_val = max(np.max(np.abs(objects)), np.max(np.abs(targets)))
    print(f"Normalization factor (max_val): {max_val:.4f}")
    
    objects_norm = objects / max_val
    targets_norm = targets / max_val

    initial_criterion_check = nn.MSELoss()
    MAE_start = initial_criterion_check(torch.from_numpy(objects_norm[:20100]).float(), torch.from_numpy(targets_norm[:20100]).float())
    print(f'Initial MSE (Normalized): {MAE_start:.6f}')

    train_dataloader, test_dataloader = data_prepare(objects_norm, targets_norm, window_size=WINDOW_SIZE, batch_size=batch_size)

    model = KAN_model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {total_params} !!!')
    
    criterion = nn.MSELoss() 
    lr = 5e-3
    num_epochs = 25
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    #weights_file = "qam_64_kan_weights.pt"

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
    MOD_ORDER = 64
    LSB = 4
    main(train_en = TRAIN_EN, lsb = LSB, mod_order = MOD_ORDER)