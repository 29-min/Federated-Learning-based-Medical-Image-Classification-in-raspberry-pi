"""
Dirichlet 기반 Non-IID 데이터 분할 + Stratified Train/Test Split
- PathMNIST의 Train 데이터(89,958개)만 사용
- 각 클라이언트에 Dirichlet으로 데이터 분배
- 클라이언트 내에서 Stratified Split으로 Train/Test 비율 유지
"""

import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
import numpy as np
import medmnist
from medmnist.info import INFO
import os
from sklearn.model_selection import train_test_split

# PathMNIST 클래스 이름
CLASS_NAMES = [
    'ADI (Adipose)',
    'BACK (Background)', 
    'DEB (Debris)',
    'LYM (Lymphocytes)',
    'MUC (Mucus)',
    'MUS (Smooth Muscle)',
    'NORM (Normal Mucosa)',
    'STR (Stroma)',
    'TUM (Tumor)'
]

def dirichlet_split_noniid(labels, num_clients, num_classes, alpha, seed=42):
    """
    Dirichlet Distribution 기반 Non-IID 데이터 분할
    """
    np.random.seed(seed)
    
    # 클래스별로 데이터 인덱스 그룹화
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # 각 클라이언트가 받을 인덱스 저장소 초기화
    client_indices = [[] for _ in range(num_clients)]
    
    # 각 클래스를 Dirichlet 분포로 클라이언트들에게 분배
    for c_idx, c_indices in enumerate(class_indices):
        np.random.shuffle(c_indices)
        
        # Dirichlet 분포로 비율 생성
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # 비율을 실제 데이터 개수로 변환
        proportions = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
        
        # 실제로 데이터 분할
        split_indices = np.split(c_indices, proportions)
        
        # 각 클라이언트에게 할당
        for client_id, indices in enumerate(split_indices):
            client_indices[client_id].extend(indices.tolist())
    
    # 각 클라이언트 내에서 데이터 섞기
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices


def stratified_train_test_split(dataset, indices, test_ratio=0.2, seed=42):
    """
    Stratified Train/Test 분할 - 클래스 비율 유지
    """
    # 해당 인덱스의 라벨 추출
    labels = []
    for idx in indices:
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            labels.append(label.item())
        elif isinstance(label, np.ndarray):
            labels.append(label.item())
        else:
            labels.append(int(label))
    
    labels = np.array(labels)
    indices = np.array(indices)
    
    try:
        # Stratified split 시도
        train_idx, test_idx = train_test_split(
            indices, 
            test_size=test_ratio,
            stratify=labels,
            random_state=seed
        )
    except ValueError as e:
        # Stratified split 실패 시 (클래스당 샘플 부족)
        print(f"  ⚠️ Stratified split 실패, random split 사용: {e}")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed
        )
    
    return train_idx.tolist(), test_idx.tolist()


def analyze_distribution(dataset, indices, name=""):
    """클래스 분포 분석 및 출력"""
    labels = []
    for idx in indices:
        _, label = dataset[idx]
        if isinstance(label, torch.Tensor):
            labels.append(label.item())
        elif isinstance(label, np.ndarray):
            labels.append(label.item())
        else:
            labels.append(int(label))
    
    labels = np.array(labels)
    
    print(f"\n클래스 분포 ({name}):")
    print(f"  {'클래스':<10} {'샘플 수':>10} {'비율':>10} {'시각화':<30}")
    print(f"  {'-'*60}")
    
    total = len(labels)
    for i in range(9):  # PathMNIST 9개 클래스
        count = np.sum(labels == i)
        ratio = count / total * 100 if total > 0 else 0
        bar = '█' * int(ratio / 2)
        print(f"  {i:<10} {count:>10} {ratio:>9.1f}% {bar}")
    
    return labels


def create_client_data(num_clients=3, alpha=0.5, test_ratio=0.2, seed=42):
    """
    클라이언트별 데이터 생성 (Stratified Split 적용)
    """
    print("="*60)
    print("Dirichlet Non-IID 데이터 분할 (Stratified Train/Test)")
    print("="*60)
    print(f"  클라이언트 수: {num_clients}")
    print(f"  Alpha: {alpha}")
    print(f"  Test 비율: {test_ratio}")
    print(f"  Seed: {seed}")
    print("="*60)
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # PathMNIST Train 데이터만 로드
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    train_dataset = DataClass(split='train', transform=transform, download=True)
    
    print(f"\n📊 PathMNIST Train 데이터: {len(train_dataset)}개")
    
    # 전체 라벨 추출
    all_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        if isinstance(label, torch.Tensor):
            all_labels.append(label.item())
        elif isinstance(label, np.ndarray):
            all_labels.append(label.item())
        else:
            all_labels.append(int(label))
    
    all_labels = np.array(all_labels)
    
    # 전체 클래스 분포 출력
    print("\n📊 전체 데이터 클래스 분포:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt}개 ({cnt/len(all_labels)*100:.1f}%)")
    
    # Dirichlet 분할
    print("\n" + "="*60)
    print("📦 Dirichlet 분포로 클라이언트별 데이터 분할...")
    print("="*60)
    
    client_indices = dirichlet_split_noniid(
        all_labels, 
        num_clients=num_clients,
        num_classes=9,
        alpha=alpha,
        seed=seed
    )
    
    # 각 클라이언트별 처리
    client_names = ['Pi5 8GB', 'Pi5 4GB', 'Pi4B 2GB']
    
    for client_id in range(num_clients):
        print("\n" + "="*60)
        print(f"클라이언트 {client_id} ({client_names[client_id]}) 데이터 처리")
        print("="*60)
        
        indices = client_indices[client_id]
        total_samples = len(indices)
        
        print(f"\n할당된 전체 데이터: {total_samples}개")
        
        # 전체 분포 확인
        client_labels = analyze_distribution(train_dataset, indices, "전체")
        
        # Stratified Train/Test 분할
        print(f"\n🔀 Stratified Train/Test 분할 (비율: {1-test_ratio:.0%}/{test_ratio:.0%})...")
        
        train_indices, test_indices = stratified_train_test_split(
            train_dataset, 
            indices, 
            test_ratio=test_ratio,
            seed=seed
        )
        
        print(f"  - Train: {len(train_indices)}개 ({len(train_indices)/total_samples*100:.1f}%)")
        print(f"  - Test: {len(test_indices)}개 ({len(test_indices)/total_samples*100:.1f}%)")
        
        # Train 분포 확인
        train_labels = analyze_distribution(train_dataset, train_indices, "Train")
        
        # Test 분포 확인
        test_labels = analyze_distribution(train_dataset, test_indices, "Test")
        
        # 분포 일치 검증
        print("\n✅ Train/Test 분포 비교:")
        print(f"  {'클래스':<10} {'Train %':>10} {'Test %':>10} {'차이':>10}")
        print(f"  {'-'*45}")
        
        for i in range(9):
            train_ratio = np.sum(train_labels == i) / len(train_labels) * 100 if len(train_labels) > 0 else 0
            test_ratio_val = np.sum(test_labels == i) / len(test_labels) * 100 if len(test_labels) > 0 else 0
            diff = abs(train_ratio - test_ratio_val)
            status = "✓" if diff < 3 else "⚠️"
            print(f"  {i:<10} {train_ratio:>9.1f}% {test_ratio_val:>9.1f}% {diff:>8.1f}% {status}")
        
        # Subset 생성
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(train_dataset, test_indices)
        
        # 저장
        save_data = {
            'train': train_subset,
            'val': test_subset,  # 기존 코드 호환성을 위해 'val' 키 사용
            'num_classes': 9,
            'train_indices': train_indices,
            'test_indices': test_indices,
            'alpha': alpha,
            'client_id': client_id
        }
        
        filename = f'client_{client_id}_data.pt'
        torch.save(save_data, filename)
        print(f"\n💾 저장 완료: {filename}")
    
    print("\n" + "="*60)
    print("✅ 모든 클라이언트 데이터 생성 완료!")
    print("="*60)
    
    # 최종 요약
    print("\n📋 최종 요약:")
    print(f"{'Client':<15} {'Total':>10} {'Train':>10} {'Test':>10}")
    print("-"*50)
    
    total_all = 0
    for client_id in range(num_clients):
        data = torch.load(f'client_{client_id}_data.pt', weights_only=False)
        train_len = len(data['train'])
        test_len = len(data['val'])
        total_len = train_len + test_len
        total_all += total_len
        print(f"Client {client_id:<8} {total_len:>10} {train_len:>10} {test_len:>10}")
    
    print("-"*50)
    print(f"{'Total':<15} {total_all:>10}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Non-IID 데이터 분할 (Stratified)')
    parser.add_argument('--num_clients', type=int, default=3, help='클라이언트 수')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='테스트 비율')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    create_client_data(
        num_clients=args.num_clients,
        alpha=args.alpha,
        test_ratio=args.test_ratio,
        seed=args.seed
    )