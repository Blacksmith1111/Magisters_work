import numpy as np
import matplotlib.pyplot as plt
import torch
from kan import KAN
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json


DEVICE = 'cuda:0'

def data_prepare(objects, targets, batch_size=64):
    print(f"objects.shape = {objects.shape}; targets.shape = {targets.shape}")
    
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
    
    dataset = {
    'train_input': torch.from_numpy(x_train).float().to(DEVICE),
    'train_label': torch.from_numpy(y_train).float().to(DEVICE),
    'test_input': torch.from_numpy(x_test).float().to(DEVICE),
    'test_label': torch.from_numpy(y_test).float().to(DEVICE)
    }

    return dataset

def load_kan1(weights_path, config_path, device):
    with open(config_path, 'r') as f:
        conf = json.load(f)
    
    model = KAN(width = conf['width'], grid = conf['grid'], k = conf['k']).to(device)
    
    model(torch.zeros((1, 2)).to(device))
    
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

def load_kan(weights_path, device):
    # Загружаем общий файл
    checkpoint = torch.load(weights_path, map_location=device)
    conf = checkpoint['config']
    
    print(f"Loading from checkpoint: grid={conf['grid']}, width={conf['width']}")
    
    # Создаем модель по конфигу из файла
    model = KAN(width=conf['width'], grid=conf['grid'], k=conf['k']).to(device)
    
    # "Прогрев" для инициализации тензоров
    model(torch.zeros((1, 2)).to(device))
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main(train_en = 1):

    objects = np.load("model_objects_64_qam.npy")[1000:1000 + 201000]
    targets = np.load("model_targets_64_qam.npy")[1000:1000 + 201000]
    objects = np.column_stack((objects.real, objects.imag))
    targets = np.column_stack((targets.real, targets.imag))

    norm_val = max(np.max(np.abs(objects)), np.max(np.abs(targets)))

    #norm_val = np.max(np.abs(objects))
    objects = objects / norm_val
    targets = targets / norm_val
    dataset = data_prepare(objects = objects, targets = targets)

    model_config = {
        "width": [2, 8, 2],
        "grid": 5, 
        "k": 3
    }
 
    
    weights_file = 'kan_weights_64qam.pt'
    config_file = 'kan_config.json'
    if train_en:
        model = KAN(width=model_config["width"], grid=model_config["grid"], k=model_config["k"]).to(DEVICE)
        
        # Пойдем чуть плавнее по сеткам
        grids = [10, 20, 30] 
        
        # 1. Сначала учим на базовой сетке (g=5)
        print("Initial training (grid 5)...")
        model.fit(dataset, opt="Adam", steps=50, lr=1e-3, update_grid=True) # Здесь можно True для старта
        
        for g in grids:
            with torch.no_grad():
                model.update_grid_from_samples(dataset['train_input'])
            
            print(f"Refining to {g}...")
            model = model.refine(g)
            
            current_lr = 1e-3 if g < 20 else 2e-4
            
            print(f"--- Training with Grid: {model.grid}, lr: {current_lr} ---")
            model.fit(dataset, opt="Adam", steps=50, lr=current_lr, update_grid=False)

        print("Final polishing with LBFGS...")
        try:
            results = model.fit(dataset, opt="LBFGS", steps=50, update_grid=False)
        except Exception as e:
            print(f"LBFGS failed with {e}, staying with Adam results")


    
        model.plot()
        ### Plot the loss graph
        plt.figure(11)
        plt.plot(results['train_loss'], label = 'Train loss')
        plt.plot(results['test_loss'], label = 'Test loss')
        plt.legend()
        plt.grid()
        plt.show()

        print(f"Final parameters: {sum(p.numel() for p in model.parameters())}")

        model.eval()
        with torch.no_grad():
            model(dataset['train_input'])

        model = model.prune(node_th = 1e-2, edge_th = 1e-2)

        print("Fine-tuning after pruning...")
        model.fit(dataset, opt="LBFGS", steps=10)

        model.plot(beta=100)
        plt.show()

        print(f"Parameters after pruning: {sum(p.numel() for p in model.parameters())}")
        
        
        #model.eval()
        
        current_grid = model.grid
        print(f"DEBUG ПЕРЕД СОХРАНЕНИЕМ: Актуальная сетка в модели = {current_grid}")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'width': model_config['width'], 
                'grid': current_grid, # Берем ПРЯМО из атрибута объекта
                'k': model.k
            }
        }
        
        final_file = 'kan_final_v1.pt' 
        torch.save(checkpoint, final_file)
            
        print('Model saved')
    else:
        #model = load_kan(weights_file, config_file, DEVICE)
        model = load_kan('kan_final_v1.pt', DEVICE)
        model.eval()
        with torch.no_grad():
            prediction = model(dataset['test_input'])
        targets_test = dataset['test_label']
        print(prediction.shape, targets_test.shape)
        criterion_test = nn.L1Loss()
        MAE_test = criterion_test(prediction, targets_test).item()
        print(f'MAE on test is {MAE_test}')
        plt.figure(100)
        plt.plot(targets_test.cpu().numpy()[:1000, 0], label = 'Targets test, real', marker = 'o')
        plt.plot(prediction.cpu().numpy()[:1000, 0], label = 'Prediction test, real', marker = 'o')
        plt.legend()
        plt.grid()

        plt.figure(101)
        plt.plot(targets_test.cpu().numpy()[:1000, 1], label = 'Targets test, imag', marker = 'o')
        plt.plot(prediction.cpu().numpy()[:1000, 1], label = 'Prediction test, imag', marker = 'o')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    TRAIN_EN = 0
    main(TRAIN_EN)