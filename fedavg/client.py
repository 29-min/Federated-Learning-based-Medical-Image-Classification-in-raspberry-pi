"""
7주차: 연합학습 클라이언트 (Dirichlet 기반 Non-IID)
PathMNIST (9개 클래스)를 Dirichlet Distribution으로 불균등 분할
현실적인 병원별 데이터 분포 차이를 시뮬레이션
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np
import argparse
import medmnist
from medmnist.info import INFO
from sklearn.metrics import f1_score

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        
        # 3채널 입력 (PathMNIST는 RGB)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # PathMNIST 28x28 이미지
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def dirichlet_split_noniid(labels, num_clients, num_classes, alpha, seed=42):
    """
    Dirichlet Distribution 기반 Non-IID 데이터 분할
    
    Args:
        labels: 전체 데이터의 라벨 배열 (numpy array)
        num_clients: 클라이언트 수
        num_classes: 클래스 수
        alpha: Dirichlet 분포 파라미터 (작을수록 더 불균등)
               - alpha=0.1: 극단적 Non-IID (각 클라이언트가 1-2개 클래스 집중)
               - alpha=0.5: 중간 Non-IID (각 클라이언트가 3-4개 클래스 위주)
               - alpha=10: 거의 균등 (IID에 가까움)
        seed: 랜덤 시드
    
    Returns:
        client_indices: 각 클라이언트가 가져갈 데이터 인덱스 리스트
    """
    np.random.seed(seed)
    
    # 1단계: 클래스별로 데이터 인덱스 그룹화
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # 각 클래스별 데이터 개수 확인
    for i, indices in enumerate(class_indices):
        print(f"    클래스 {i}: {len(indices)}개 샘플")
    
    # 2단계: 각 클라이언트가 받을 인덱스 저장소 초기화
    client_indices = [[] for _ in range(num_clients)]
    
    # 3단계: 각 클래스를 Dirichlet 분포로 클라이언트들에게 분배
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
    
    # 4단계: 각 클라이언트 내에서 데이터 섞기 (학습 효율성)
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])
    
    return client_indices

def analyze_class_distribution(dataset, indices):
    """클래스 분포 분석 및 시각화"""
    labels = []
    for idx in indices:
        _, label = dataset[idx]
        labels.append(label.item() if torch.is_tensor(label) else label)
    
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    return distribution

def load_data_dirichlet(client_id, total_clients, alpha=0.5, batch_size=32, seed=42):
    """
    Dirichlet 분포 기반 Non-IID 데이터 로드
    
    Args:
        client_id: 현재 클라이언트 ID
        total_clients: 전체 클라이언트 수
        alpha: Dirichlet 파라미터 (Non-IID 강도)
        batch_size: 배치 크기
        seed: 랜덤 시드
    """
    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # PathMNIST 로드 (9개 클래스)
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    num_classes = len(info['label'])
    
    print(f"\n{'='*60}")
    print(f"PathMNIST 정보:")
    print(f"  - 클래스 수: {num_classes}")
    print(f"  - 설명: {info['description']}")
    print(f"{'='*60}\n")
    
    DataClass = getattr(medmnist, info['python_class'])
    
    # 전체 데이터셋 다운로드
    train_dataset = DataClass(split='train', transform=transform, download=True)
    test_dataset = DataClass(split='test', transform=transform, download=True)
    
    # 라벨 추출 (numpy array로 변환)
    train_labels = np.array([label.item() for _, label in train_dataset])
    test_labels = np.array([label.item() for _, label in test_dataset])
    
    print(f"전체 데이터셋 크기:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    print(f"\nDirichlet 분할 시작 (alpha={alpha})...")
    print(f"  ※ alpha가 작을수록 클래스 분포가 불균등해집니다")
    print(f"\nTrain 데이터 클래스별 샘플 수:")
    
    # Dirichlet 분할 수행
    train_client_indices = dirichlet_split_noniid(
        train_labels, 
        total_clients, 
        num_classes, 
        alpha, 
        seed
    )
    
    print(f"\nTest 데이터 클래스별 샘플 수:")
    test_client_indices = dirichlet_split_noniid(
        test_labels, 
        total_clients, 
        num_classes, 
        alpha, 
        seed + 1  # 다른 시드로 독립적 분할
    )
    
    # 현재 클라이언트의 인덱스 가져오기
    client_train_indices = train_client_indices[client_id]
    client_test_indices = test_client_indices[client_id]
    
    # 클래스 분포 분석
    train_distribution = analyze_class_distribution(train_dataset, client_train_indices)
    test_distribution = analyze_class_distribution(test_dataset, client_test_indices)
    
    # Subset 생성
    train_subset = Subset(train_dataset, client_train_indices)
    test_subset = Subset(test_dataset, client_test_indices)
    
    # DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 상세 정보 출력
    print(f"\n{'='*60}")
    print(f"클라이언트 {client_id} 데이터 할당 완료")
    print(f"{'='*60}")
    print(f"\n할당된 데이터:")
    print(f"  - Train: {len(train_subset)} samples ({len(train_subset)/len(train_dataset)*100:.1f}%)")
    print(f"  - Test: {len(test_subset)} samples ({len(test_subset)/len(test_dataset)*100:.1f}%)")
    
    print(f"\n클래스 분포 (Train):")
    print(f"  {'클래스':<8} {'샘플 수':<10} {'비율':<10} {'시각화'}")
    print(f"  {'-'*50}")
    for label in range(num_classes):
        count = train_distribution.get(label, 0)
        percentage = count / len(train_subset) * 100
        bar = '█' * int(percentage / 2)  # 50%당 25칸
        print(f"  {label:<8} {count:<10} {percentage:>5.1f}%     {bar}")
    
    print(f"\n클래스 분포 (Test):")
    print(f"  {'클래스':<8} {'샘플 수':<10} {'비율':<10} {'시각화'}")
    print(f"  {'-'*50}")
    for label in range(num_classes):
        count = test_distribution.get(label, 0)
        percentage = count / len(test_subset) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {label:<8} {count:<10} {percentage:>5.1f}%     {bar}")
    
    print(f"{'='*60}\n")
    
    return train_loader, test_loader, num_classes

def calculate_class_weights(train_loader, num_classes=9):
    """클래스 가중치 계산"""
    class_counts = torch.zeros(num_classes)
    
    for _, labels in train_loader:
        labels = labels.squeeze()
        for label in labels:
            class_counts[label.item()] += 1
    
    # 역수로 가중치 계산 (희귀 클래스에 높은 가중치)
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    
    # 0개인 클래스는 가중치 0 (학습 안 함)
    class_weights[class_counts == 0] = 0.0
    
    return class_weights

def train(model, train_loader, epochs=5, lr=0.01):
    """로컬 학습 (클래스 가중치 적용, F1 score 계산)"""
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(train_loader).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # F1 score 계산을 위한 예측 및 실제 라벨 저장
        all_predictions = []
        all_targets = []
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).squeeze().long()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # F1 score 계산을 위해 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # F1 score 계산 (macro average)
        epoch_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, F1={epoch_f1:.4f}")
    
    avg_loss = running_loss / len(train_loader)
    avg_acc = 100. * correct / total
    avg_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    return avg_loss, avg_acc, avg_f1

def test(model, test_loader):
    """로컬 평가 (F1 score 포함)"""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    
    # F1 score 계산을 위한 예측 및 실제 라벨 저장
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze().long()
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # F1 score 계산을 위해 저장
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    
    # F1 score 계산 (macro average)
    test_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    return test_loss, test_acc, test_f1

class FlowerClient(fl.client.NumPyClient):
    """Flower 클라이언트 (체크포인트 기능 추가)"""
    
    def __init__(self, client_id, model, train_loader, test_loader, local_epochs, lr):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.checkpoint_dir = f"checkpoints/client_{client_id}"
        
        # 체크포인트 디렉토리 생성
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 마지막 라운드 번호 추적
        self.current_round = 0
        
        # 체크포인트 로드 시도
        self.load_checkpoint()
    
    def save_checkpoint(self):
        """현재 모델 상태 저장"""
        checkpoint_path = f"{self.checkpoint_dir}/round_{self.current_round}.pt"
        torch.save({
            'round': self.current_round,
            'model_state_dict': self.model.state_dict(),
        }, checkpoint_path)
        print(f"💾 체크포인트 저장: {checkpoint_path}")
    
    def load_checkpoint(self):
        """가장 최근 체크포인트 로드"""
        import os
        import glob
        
        # 저장된 체크포인트 찾기
        checkpoints = glob.glob(f"{self.checkpoint_dir}/round_*.pt")
        
        if checkpoints:
            # 가장 최근 라운드 찾기
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            checkpoint = torch.load(latest_checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_round = checkpoint['round']
            
            print(f"✅ 체크포인트 로드: {latest_checkpoint}")
            print(f"   라운드 {self.current_round}부터 재개합니다.")
        else:
            print(f"🆕 새로운 학습 시작")
    
    def get_parameters(self, config):
        """모델 파라미터 반환"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """서버 파라미터 설정"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """로컬 학습"""
        self.current_round += 1
        
        print(f"\n{'='*60}")
        print(f"클라이언트 {self.client_id} - 라운드 {self.current_round} 학습 시작")
        print(f"{'='*60}")
        
        self.set_parameters(parameters)
        
        train_loss, train_acc, train_f1 = train(
            self.model,
            self.train_loader,
            epochs=self.local_epochs,
            lr=self.lr
        )
        
        # 학습 완료 후 체크포인트 저장
        self.save_checkpoint()
        
        print(f"학습 완료: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, F1={train_f1:.4f}")
        print(f"{'='*60}\n")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"f1_score": train_f1}
    
    def evaluate(self, parameters, config):
        """로컬 평가"""
        print(f"\n클라이언트 {self.client_id} - 라운드 {self.current_round} 평가 시작")
        
        self.set_parameters(parameters)
        
        test_loss, test_acc, test_f1 = test(self.model, self.test_loader)
        
        print(f"평가 완료: Loss={test_loss:.4f}, Acc={test_acc:.2f}%, F1={test_f1:.4f}\n")
        
        return test_loss, len(self.test_loader.dataset), {"accuracy": test_acc, "f1_score": test_f1}

def main():
    """클라이언트 메인 함수"""
    parser = argparse.ArgumentParser(description='연합학습 클라이언트 (Dirichlet Non-IID)')
    parser.add_argument('--client_id', type=int, required=True, help='클라이언트 ID')
    parser.add_argument('--total_clients', type=int, default=3, help='전체 클라이언트 수')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha (작을수록 Non-IID)')
    parser.add_argument('--server_address', type=str, default='192.168.45.100:8080', help='서버 주소')
    parser.add_argument('--local_epochs', type=int, default=5, help='로컬 에포크')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.01, help='학습률')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"연합학습 클라이언트 {args.client_id} (Dirichlet Non-IID)")
    print("="*60)
    print(f"서버: {args.server_address}")
    print(f"전체 클라이언트: {args.total_clients}")
    print(f"모델: SimpleCNN")
    print(f"Alpha (Non-IID 강도): {args.alpha}")
    print(f"  ※ alpha=0.1: 극단적 Non-IID")
    print(f"  ※ alpha=0.5: 중간 Non-IID")
    print(f"  ※ alpha=10: 거의 IID")
    print(f"로컬 에포크: {args.local_epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"학습률: {args.lr}")
    print("="*60)
    
    # 시드 설정 (재현성)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 데이터 로드 (Dirichlet 분할)
    train_loader, test_loader, num_classes = load_data_dirichlet(
        args.client_id,
        args.total_clients,
        args.alpha,
        args.batch_size
    )
    
    # 모델 생성
    model = SimpleCNN(num_classes=num_classes).to(device)
    print(f"\n🔧 SimpleCNN 모델 생성 완료")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   총 파라미터 수: {total_params:,}")
    
    # Flower 클라이언트 생성
    client = FlowerClient(
        args.client_id,
        model,
        train_loader,
        test_loader,
        args.local_epochs,
        args.lr
    )
    
    print(f"\n서버 {args.server_address}에 연결 중...")
    
    # 서버 연결
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )
    
    print(f"\n클라이언트 {args.client_id} 종료")

if __name__ == "__main__":
    main()