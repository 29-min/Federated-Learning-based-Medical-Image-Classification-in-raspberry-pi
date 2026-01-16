"""
IID 데이터 전처리 스크립트
- 모든 클라이언트에 클래스를 균등하게 분배
- Non-IID와 비교를 위한 베이스라인
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import medmnist
from medmnist.info import INFO
from torch.utils.data import Subset

# 시드 고정
torch.manual_seed(42)
np.random.seed(42)

def create_iid_split(dataset, num_clients=3):
    """
    IID 방식으로 데이터를 균등하게 분할
    각 클라이언트가 모든 클래스를 비슷한 비율로 보유
    """
    total_samples = len(dataset)
    indices = np.random.permutation(total_samples)
    
    # 단순 균등 분할
    split_size = total_samples // num_clients
    client_indices = []
    
    for i in range(num_clients):
        start_idx = i * split_size
        if i == num_clients - 1:
            # 마지막 클라이언트는 나머지 모두 포함
            end_idx = total_samples
        else:
            end_idx = (i + 1) * split_size
        
        client_indices.append(indices[start_idx:end_idx])
    
    return client_indices

def analyze_distribution(subset, num_classes=9):
    """클라이언트의 클래스 분포 분석"""
    class_counts = np.zeros(num_classes)
    
    for idx in range(len(subset)):
        _, label = subset[idx]
        label = label.item() if torch.is_tensor(label) else label
        class_counts[label] += 1
    
    return class_counts

def main():
    print("="*70)
    print("IID 데이터 전처리 시작")
    print("="*70)
    
    # PathMNIST 데이터셋 로드
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    num_classes = len(info['label'])
    
    print(f"\n📊 데이터셋 정보:")
    print(f"  클래스 수: {num_classes}")
    
    # Train/Val 데이터셋 로드
    train_dataset = DataClass(split='train', transform=transform, download=True)
    val_dataset = DataClass(split='val', transform=transform, download=True)
    
    print(f"  Train 샘플: {len(train_dataset)}")
    print(f"  Val 샘플: {len(val_dataset)}")
    
    # IID 분할
    num_clients = 3
    print(f"\n🔄 IID 방식으로 {num_clients}개 클라이언트에 분할 중...")
    
    train_client_indices = create_iid_split(train_dataset, num_clients)
    val_client_indices = create_iid_split(val_dataset, num_clients)
    
    # 각 클라이언트 데이터 저장
    for client_id in range(num_clients):
        print(f"\n{'='*70}")
        print(f"클라이언트 {client_id} 데이터 생성")
        print(f"{'='*70}")
        
        # Subset 생성
        train_subset = Subset(train_dataset, train_client_indices[client_id])
        val_subset = Subset(val_dataset, val_client_indices[client_id])
        
        print(f"  Train 샘플: {len(train_subset)}")
        print(f"  Val 샘플: {len(val_subset)}")
        
        # 클래스 분포 분석
        train_dist = analyze_distribution(train_subset, num_classes)
        val_dist = analyze_distribution(val_subset, num_classes)
        
        print(f"\n  Train 클래스 분포:")
        for class_idx in range(num_classes):
            percentage = (train_dist[class_idx] / len(train_subset)) * 100
            print(f"    Class {class_idx}: {int(train_dist[class_idx]):5d} ({percentage:5.2f}%)")
        
        print(f"\n  Val 클래스 분포:")
        for class_idx in range(num_classes):
            percentage = (val_dist[class_idx] / len(val_subset)) * 100
            print(f"    Class {class_idx}: {int(val_dist[class_idx]):5d} ({percentage:5.2f}%)")
        
        # 데이터 저장
        save_data = {
            'train': train_subset,
            'val': val_subset,
            'num_classes': num_classes,
            'train_distribution': train_dist,
            'val_distribution': val_dist
        }
        
        filename = f'client_{client_id}_data_iid.pt'
        torch.save(save_data, filename)
        print(f"\n  ✅ 저장 완료: {filename}")
    
    # 전체 분포 요약
    print(f"\n{'='*70}")
    print("전체 데이터 분포 요약")
    print(f"{'='*70}")
    
    total_train_dist = np.zeros(num_classes)
    total_val_dist = np.zeros(num_classes)
    
    for client_id in range(num_clients):
        data = torch.load(f'client_{client_id}_data_iid.pt', weights_only=False)
        total_train_dist += data['train_distribution']
        total_val_dist += data['val_distribution']
    
    print("\n📊 Train 전체 분포:")
    for class_idx in range(num_classes):
        print(f"  Class {class_idx}: {int(total_train_dist[class_idx]):5d}")
    
    print("\n📊 Val 전체 분포:")
    for class_idx in range(num_classes):
        print(f"  Class {class_idx}: {int(total_val_dist[class_idx]):5d}")
    
    print(f"\n{'='*70}")
    print("✅ IID 데이터 전처리 완료!")
    print(f"{'='*70}")
    print("\n생성된 파일:")
    for client_id in range(num_clients):
        print(f"  - client_{client_id}_data_iid.pt")
    print("\n다음 단계: 클라이언트 코드에서 '_iid.pt' 파일을 로드하도록 수정")

if __name__ == "__main__":
    main()