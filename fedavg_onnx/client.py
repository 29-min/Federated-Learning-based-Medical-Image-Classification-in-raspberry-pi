"""
ONNX vs PyTorch 비교 클라이언트
- 각 라운드마다 PyTorch와 ONNX 양쪽으로 추론
- 추론 시간, 정확도, F1, 모델 크기 모두 측정하여 서버로 전송
"""

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from sklearn.metrics import f1_score
import os
import time
import tracemalloc
import gc

# ONNX
import torch.onnx
import onnx
from onnxsim import simplify
import onnxruntime as ort

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------
# SimpleCNN 모델
# ------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
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

# ------------------------------------------------------------------------
# ONNX Manager (모델 크기 측정 추가)
# ------------------------------------------------------------------------
class ONNXManager:
    def __init__(self, model, client_id):
        self.pytorch_model = model
        self.client_id = client_id
        self.onnx_path = f'model_client_{client_id}.onnx'
        self.simplified_path = f'model_client_{client_id}_simplified.onnx'
        self.pytorch_path = f'model_client_{client_id}.pth'
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        
        # 모델 크기 저장
        self.pytorch_size_kb = 0
        self.onnx_original_size_kb = 0
        self.onnx_simplified_size_kb = 0
    
    def get_pytorch_size(self):
        """PyTorch 모델 크기 측정"""
        torch.save(self.pytorch_model.state_dict(), self.pytorch_path)
        size_bytes = os.path.getsize(self.pytorch_path)
        self.pytorch_size_kb = size_bytes / 1024
        return self.pytorch_size_kb
    
    def export_and_simplify(self):
        """ONNX 변환 및 Simplify, 크기 측정"""
        self.pytorch_model.eval()
        dummy_input = torch.randn(1, 3, 28, 28).to(device)
        
        # PyTorch 모델 크기 측정
        self.get_pytorch_size()
        
        # ONNX Export
        torch.onnx.export(
            self.pytorch_model,
            dummy_input,
            self.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # ONNX 원본 크기
        self.onnx_original_size_kb = os.path.getsize(self.onnx_path) / 1024
        
        # Simplify
        simplified_success = False
        try:
            onnx_model = onnx.load(self.onnx_path)
            simplified_model, check = simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, self.simplified_path)
                self.onnx_simplified_size_kb = os.path.getsize(self.simplified_path) / 1024
                simplified_success = True
            else:
                self.onnx_simplified_size_kb = self.onnx_original_size_kb
        except Exception as e:
            print(f"  ⚠️ Simplifier 오류: {e}")
            self.onnx_simplified_size_kb = self.onnx_original_size_kb
        
        return self.simplified_path if simplified_success else self.onnx_path
    
    def get_size_comparison(self):
        """모델 크기 비교 결과 반환"""
        reduction_from_pytorch = ((self.pytorch_size_kb - self.onnx_simplified_size_kb) 
                                   / self.pytorch_size_kb * 100) if self.pytorch_size_kb > 0 else 0
        reduction_from_original = ((self.onnx_original_size_kb - self.onnx_simplified_size_kb) 
                                    / self.onnx_original_size_kb * 100) if self.onnx_original_size_kb > 0 else 0
        
        return {
            'pytorch_size_kb': self.pytorch_size_kb,
            'onnx_original_size_kb': self.onnx_original_size_kb,
            'onnx_simplified_size_kb': self.onnx_simplified_size_kb,
            'reduction_from_pytorch_pct': reduction_from_pytorch,
            'reduction_from_original_pct': reduction_from_original
        }
    
    def load_session(self, onnx_path):
        self.ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
    
    def predict(self, data):
        if self.ort_session is None:
            raise RuntimeError("ONNX session not loaded")
        
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        outputs = self.ort_session.run(
            [self.output_name], 
            {self.input_name: data.astype(np.float32)}
        )
        
        return outputs[0]
    
    def cleanup(self):
        """임시 파일 정리"""
        for path in [self.pytorch_path, self.onnx_path, self.simplified_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

# ------------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------------
def load_preprocessed_data(client_id, batch_size):
    file_path = f'client_{client_id}_data.pt'
    
    if not os.path.exists(file_path):
        print(f"❌ 오류: {file_path} 파일을 찾을 수 없습니다.")
        exit(1)
        
    print(f"📂 파일 로드: {file_path}")
    
    data = torch.load(file_path, map_location=device, weights_only=False)
    train_subset = data['train']
    val_subset = data['val']
    num_classes = data['num_classes']

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ 데이터 로드 완료: Train {len(train_subset)}, Val {len(val_subset)}")
    
    return train_loader, val_loader, num_classes

# ------------------------------------------------------------------------
# 클래스 가중치
# ------------------------------------------------------------------------
def calculate_class_weights(train_loader, num_classes=9):
    class_counts = torch.zeros(num_classes)
    
    if len(train_loader.dataset) == 0:
        return torch.ones(num_classes)
    
    for _, labels in train_loader:
        labels = labels.squeeze()
        for label in labels:
            class_counts[label.item()] += 1
    
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    class_weights[class_counts == 0] = 0.0
    
    return class_weights

# ------------------------------------------------------------------------
# 학습
# ------------------------------------------------------------------------
def train(model, train_loader, epochs, lr):
    class_weights = calculate_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device).squeeze().long()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
    
    return running_loss / len(train_loader)

# ------------------------------------------------------------------------
# PyTorch 추론 (tracemalloc 피크 메모리 측정)
# ------------------------------------------------------------------------
def test_pytorch(model, test_loader):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # 가비지 컬렉션 및 메모리 초기화
    gc.collect()
    
    # tracemalloc으로 피크 메모리 측정
    tracemalloc.start()
    tracemalloc.reset_peak()
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.squeeze().long()
            
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.numpy())
    
    total_time = time.time() - start_time
    
    # 피크 메모리 측정 (MB 단위)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024
    
    correct = sum([1 for p, t in zip(all_predictions, all_targets) if p == t])
    accuracy = 100. * correct / len(all_targets)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    return accuracy, f1, total_time, peak_memory_mb

# ------------------------------------------------------------------------
# ONNX 추론 (tracemalloc 피크 메모리 측정)
# ------------------------------------------------------------------------
def test_onnx(onnx_manager, test_loader):
    all_predictions = []
    all_targets = []
    
    # 가비지 컬렉션 및 메모리 초기화
    gc.collect()
    
    # tracemalloc으로 피크 메모리 측정
    tracemalloc.start()
    tracemalloc.reset_peak()
    
    start_time = time.time()
    
    for data, target in test_loader:
        target = target.squeeze().long()
        
        outputs = onnx_manager.predict(data)
        predicted = np.argmax(outputs, axis=1)
        
        all_predictions.extend(predicted)
        all_targets.extend(target.numpy())
    
    total_time = time.time() - start_time
    
    # 피크 메모리 측정 (MB 단위)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024
    
    correct = sum([1 for p, t in zip(all_predictions, all_targets) if p == t])
    accuracy = 100. * correct / len(all_targets)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
    return accuracy, f1, total_time, peak_memory_mb

# ------------------------------------------------------------------------
# Flower Client
# ------------------------------------------------------------------------
class ComparisonClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, val_loader, local_epochs, lr):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.lr = lr
        self.current_round = 0
        self.onnx_manager = ONNXManager(model, client_id)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.current_round += 1
        
        print(f"\n{'='*60}")
        print(f"클라이언트 {self.client_id} - 라운드 {self.current_round}")
        print(f"{'='*60}")
        
        self.set_parameters(parameters)
        
        if len(self.train_loader.dataset) == 0:
            return self.get_parameters(config={}), 0, {}
        
        print("🔥 PyTorch 학습 중...")
        train_loss = train(self.model, self.train_loader, self.local_epochs, self.lr)
        
        print(f"✅ 학습 완료: Loss={train_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        print(f"\n{'='*60}")
        print(f"클라이언트 {self.client_id} - 라운드 {self.current_round} 비교 평가")
        print(f"{'='*60}")
        
        self.set_parameters(parameters)
        
        if len(self.val_loader.dataset) == 0:
            return 0.0, 0, {}
        
        # 1. PyTorch 추론
        print("⚡ PyTorch 추론 중...")
        pt_acc, pt_f1, pt_time, pt_mem = test_pytorch(self.model, self.val_loader)
        print(f"  PyTorch - Acc: {pt_acc:.2f}%, F1: {pt_f1:.4f}, Time: {pt_time:.3f}초, Mem: {pt_mem:.2f}MB")
        
        # 2. ONNX 변환 및 추론
        print("🔄 ONNX 변환 중...")
        onnx_path = self.onnx_manager.export_and_simplify()
        self.onnx_manager.load_session(onnx_path)
        
        # 모델 크기 출력
        size_info = self.onnx_manager.get_size_comparison()
        print(f"\n📦 모델 크기 비교:")
        print(f"  PyTorch (.pth):      {size_info['pytorch_size_kb']:.2f} KB")
        print(f"  ONNX (원본):         {size_info['onnx_original_size_kb']:.2f} KB")
        print(f"  ONNX (Simplified):   {size_info['onnx_simplified_size_kb']:.2f} KB")
        print(f"  PyTorch 대비 절감:   {size_info['reduction_from_pytorch_pct']:.1f}%")
        print(f"  ONNX 원본 대비 절감: {size_info['reduction_from_original_pct']:.1f}%")
        
        print("\n⚡ ONNX 추론 중...")
        onnx_acc, onnx_f1, onnx_time, onnx_mem = test_onnx(self.onnx_manager, self.val_loader)
        print(f"  ONNX - Acc: {onnx_acc:.2f}%, F1: {onnx_f1:.4f}, Time: {onnx_time:.3f}초, Mem: {onnx_mem:.2f}MB")
        
        # 비교
        speedup = pt_time / onnx_time if onnx_time > 0 else 0
        acc_diff = abs(pt_acc - onnx_acc)
        mem_saving = pt_mem - onnx_mem
        mem_saving_pct = (mem_saving / pt_mem * 100) if pt_mem > 0 else 0
        
        print(f"\n📊 성능 비교:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  정확도 차이: {acc_diff:.4f}%")
        print(f"  메모리 절감: {mem_saving:.2f}MB ({mem_saving_pct:.1f}%)")
        print(f"{'='*60}\n")
        
        # 결과 반환 (모델 크기 정보 포함)
        return 0.0, len(self.val_loader.dataset), {
            'client_id': self.client_id,
            'framework': 'PyTorch',
            'accuracy': pt_acc,
            'f1_score': pt_f1,
            'inference_time': pt_time,
            'memory_usage': pt_mem,
            'onnx_accuracy': onnx_acc,
            'onnx_f1_score': onnx_f1,
            'onnx_inference_time': onnx_time,
            'onnx_memory_usage': onnx_mem,
            'speedup': speedup,
            'accuracy_diff': acc_diff,
            'memory_saving': mem_saving,
            'memory_saving_pct': mem_saving_pct,
            # 모델 크기 정보 추가
            'pytorch_size_kb': size_info['pytorch_size_kb'],
            'onnx_original_size_kb': size_info['onnx_original_size_kb'],
            'onnx_simplified_size_kb': size_info['onnx_simplified_size_kb'],
            'size_reduction_from_pytorch_pct': size_info['reduction_from_pytorch_pct'],
            'size_reduction_from_original_pct': size_info['reduction_from_original_pct']
        }

# ------------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='ONNX vs PyTorch 비교 클라이언트')
    parser.add_argument('--client_id', type=int, required=True)
    parser.add_argument('--server_address', type=str, default='192.168.45.100:8080')
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print(f"ONNX vs PyTorch 비교 클라이언트 {args.client_id}")
    print("="*60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_loader, val_loader, num_classes = load_preprocessed_data(
        args.client_id, args.batch_size
    )
    
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    client = ComparisonClient(
        args.client_id, model, train_loader, val_loader,
        args.local_epochs, args.lr
    )
    
    print(f"\n🔗 서버 {args.server_address}에 연결 중...\n")
    
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )
    
    print(f"\n클라이언트 {args.client_id} 종료")

if __name__ == "__main__":
    main()