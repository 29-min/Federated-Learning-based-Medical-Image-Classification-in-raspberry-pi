"""
ONNX vs PyTorch 비교 서버
- 서버에서도 PyTorch vs ONNX 추론 시간 비교
- 모델 크기 비교 포함
- 비교 결과를 CSV/JSON으로 저장
"""

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import argparse
from sklearn.metrics import f1_score, classification_report
import medmnist
from medmnist.info import INFO
import csv
import json
from datetime import datetime
import os
import time
import tracemalloc
import gc

# ONNX
import torch.onnx
import onnx
import onnxruntime as ort

# onnxsim은 선택적 사용 (macOS M1에서 segfault 발생 가능)
try:
    from onnxsim import simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    print("⚠️ onnxsim 미설치 - Simplify 건너뜀")

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
# 서버용 ONNX Manager
# ------------------------------------------------------------------------
class ServerONNXManager:
    def __init__(self, model, save_dir):
        self.pytorch_model = model
        self.save_dir = save_dir
        self.onnx_path = os.path.join(save_dir, 'server_model.onnx')
        self.simplified_path = os.path.join(save_dir, 'server_model_simplified.onnx')
        self.pytorch_path = os.path.join(save_dir, 'server_model.pth')
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        
        # 모델 크기
        self.pytorch_size_kb = 0
        self.onnx_original_size_kb = 0
        self.onnx_simplified_size_kb = 0
    
    def export_and_simplify(self, use_simplifier=False):
        """ONNX 변환 및 Simplify, 크기 측정
        
        Args:
            use_simplifier: Simplifier 사용 여부 (macOS M1에서는 False 권장)
        """
        self.pytorch_model.eval()
        dummy_input = torch.randn(1, 3, 28, 28).to(device)
        
        # PyTorch 모델 크기 측정
        torch.save(self.pytorch_model.state_dict(), self.pytorch_path)
        self.pytorch_size_kb = os.path.getsize(self.pytorch_path) / 1024
        
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
        
        # Simplify (선택적)
        simplified_success = False
        if use_simplifier and ONNXSIM_AVAILABLE:
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
        else:
            # Simplifier 미사용 시 원본 크기로 설정
            self.onnx_simplified_size_kb = self.onnx_original_size_kb
            if not use_simplifier:
                print("  ℹ️ Simplifier 미사용 (서버)")
        
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

# ------------------------------------------------------------------------
# 글로벌 테스트 데이터 로드
# ------------------------------------------------------------------------
def load_global_test_data(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    info = INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    test_dataset = DataClass(split='test', transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✅ 글로벌 테스트 데이터 로드: {len(test_dataset)}개")
    return test_loader

# ------------------------------------------------------------------------
# PyTorch 글로벌 평가 (tracemalloc 피크 메모리 측정)
# ------------------------------------------------------------------------
def evaluate_pytorch(model, test_loader, num_classes=9):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
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
            target = target.to(device).squeeze().long()
            
            output = model(data)
            test_loss += criterion(output, target).item()
            
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    inference_time = time.time() - start_time
    
    # 피크 메모리 측정 (MB 단위)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    per_class_accuracy = []
    for class_idx in range(num_classes):
        class_mask = [t == class_idx for t in all_targets]
        if sum(class_mask) > 0:
            class_correct = sum([1 for p, t, m in zip(all_predictions, all_targets, class_mask) if m and p == t])
            class_acc = 100. * class_correct / sum(class_mask)
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)
    
    return {
        'framework': 'PyTorch',
        'loss': test_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1,
        'per_class_accuracy': per_class_accuracy,
        'inference_time': inference_time,
        'memory_usage': peak_memory_mb
    }

# ------------------------------------------------------------------------
# ONNX 글로벌 평가 (tracemalloc 피크 메모리 측정)
# ------------------------------------------------------------------------
def evaluate_onnx(onnx_manager, test_loader, num_classes=9):
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
    
    inference_time = time.time() - start_time
    
    # 피크 메모리 측정 (MB 단위)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024
    
    correct = sum([1 for p, t in zip(all_predictions, all_targets) if p == t])
    accuracy = 100. * correct / len(all_targets)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    
    per_class_accuracy = []
    for class_idx in range(num_classes):
        class_mask = [t == class_idx for t in all_targets]
        if sum(class_mask) > 0:
            class_correct = sum([1 for p, t, m in zip(all_predictions, all_targets, class_mask) if m and p == t])
            class_acc = 100. * class_correct / sum(class_mask)
            per_class_accuracy.append(class_acc)
        else:
            per_class_accuracy.append(0.0)
    
    return {
        'framework': 'ONNX',
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1,
        'per_class_accuracy': per_class_accuracy,
        'inference_time': inference_time,
        'memory_usage': peak_memory_mb
    }

# ------------------------------------------------------------------------
# 비교 결과 로거 (확장)
# ------------------------------------------------------------------------
class ComparisonLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"comparison_{experiment_name}_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # CSV 파일들
        self.client_csv_file = f"{self.results_dir}/client_comparison_results.csv"
        self.server_csv_file = f"{self.results_dir}/server_comparison_results.csv"
        self.model_size_csv_file = f"{self.results_dir}/model_size_comparison.csv"
        
        self.init_csvs()
        
        # JSON 파일
        self.json_file = f"{self.results_dir}/detailed_comparison.json"
        self.results = {
            'client_results': [],
            'server_results': [],
            'model_sizes': []
        }
        
        print(f"📁 비교 결과 저장: {self.results_dir}")
    
    def init_csvs(self):
        # 클라이언트 결과 CSV
        client_headers = ['Round', 'Client_ID', 
                          'PT_Accuracy', 'PT_F1', 'PT_Time', 'PT_Memory',
                          'ONNX_Accuracy', 'ONNX_F1', 'ONNX_Time', 'ONNX_Memory',
                          'Speedup', 'Accuracy_Diff']
        with open(self.client_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(client_headers)
        
        # 서버 결과 CSV
        server_headers = ['Round', 
                          'PT_Accuracy', 'PT_F1', 'PT_Time', 'PT_Memory',
                          'ONNX_Accuracy', 'ONNX_F1', 'ONNX_Time', 'ONNX_Memory',
                          'Speedup', 'Accuracy_Diff']
        with open(self.server_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(server_headers)
        
        # 모델 크기 CSV
        size_headers = ['Round', 'Source', 'PyTorch_KB', 'ONNX_Original_KB', 
                        'ONNX_Simplified_KB', 'Reduction_From_PyTorch_Pct', 
                        'Reduction_From_Original_Pct']
        with open(self.model_size_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(size_headers)
    
    def log_client_result(self, round_num, metrics):
        """클라이언트 결과 기록"""
        row = [
            round_num,
            metrics.get('client_id', 0),
            f"{metrics.get('accuracy', 0):.2f}",
            f"{metrics.get('f1_score', 0):.4f}",
            f"{metrics.get('inference_time', 0):.3f}",
            f"{metrics.get('memory_usage', 0):.2f}",
            f"{metrics.get('onnx_accuracy', 0):.2f}",
            f"{metrics.get('onnx_f1_score', 0):.4f}",
            f"{metrics.get('onnx_inference_time', 0):.3f}",
            f"{metrics.get('onnx_memory_usage', 0):.2f}",
            f"{metrics.get('speedup', 0):.2f}",
            f"{metrics.get('accuracy_diff', 0):.4f}"
        ]
        
        with open(self.client_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # 모델 크기도 기록
        if 'pytorch_size_kb' in metrics:
            size_row = [
                round_num,
                f"Client_{metrics.get('client_id', 0)}",
                f"{metrics.get('pytorch_size_kb', 0):.2f}",
                f"{metrics.get('onnx_original_size_kb', 0):.2f}",
                f"{metrics.get('onnx_simplified_size_kb', 0):.2f}",
                f"{metrics.get('size_reduction_from_pytorch_pct', 0):.1f}",
                f"{metrics.get('size_reduction_from_original_pct', 0):.1f}"
            ]
            with open(self.model_size_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(size_row)
        
        self.results['client_results'].append({
            'round': round_num,
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v 
               for k, v in metrics.items()}
        })
        self._save_json()
    
    def log_server_result(self, round_num, pt_results, onnx_results, size_info):
        """서버 결과 기록"""
        speedup = pt_results['inference_time'] / onnx_results['inference_time'] if onnx_results['inference_time'] > 0 else 0
        acc_diff = abs(pt_results['accuracy'] - onnx_results['accuracy'])
        
        row = [
            round_num,
            f"{pt_results['accuracy']:.2f}",
            f"{pt_results['macro_f1']:.4f}",
            f"{pt_results['inference_time']:.3f}",
            f"{pt_results['memory_usage']:.2f}",
            f"{onnx_results['accuracy']:.2f}",
            f"{onnx_results['macro_f1']:.4f}",
            f"{onnx_results['inference_time']:.3f}",
            f"{onnx_results['memory_usage']:.2f}",
            f"{speedup:.2f}",
            f"{acc_diff:.4f}"
        ]
        
        with open(self.server_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # 모델 크기 기록
        size_row = [
            round_num,
            "Server",
            f"{size_info['pytorch_size_kb']:.2f}",
            f"{size_info['onnx_original_size_kb']:.2f}",
            f"{size_info['onnx_simplified_size_kb']:.2f}",
            f"{size_info['reduction_from_pytorch_pct']:.1f}",
            f"{size_info['reduction_from_original_pct']:.1f}"
        ]
        with open(self.model_size_csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(size_row)
        
        self.results['server_results'].append({
            'round': round_num,
            'pytorch': {
                'accuracy': float(pt_results['accuracy']),
                'macro_f1': float(pt_results['macro_f1']),
                'inference_time': float(pt_results['inference_time']),
                'memory_usage': float(pt_results['memory_usage']),
                'per_class_f1': pt_results['per_class_f1'].tolist(),
                'per_class_accuracy': pt_results['per_class_accuracy']
            },
            'onnx': {
                'accuracy': float(onnx_results['accuracy']),
                'macro_f1': float(onnx_results['macro_f1']),
                'inference_time': float(onnx_results['inference_time']),
                'memory_usage': float(onnx_results['memory_usage']),
                'per_class_f1': onnx_results['per_class_f1'].tolist(),
                'per_class_accuracy': onnx_results['per_class_accuracy']
            },
            'speedup': float(speedup),
            'accuracy_diff': float(acc_diff),
            'model_size': size_info
        })
        
        self.results['model_sizes'].append({
            'round': round_num,
            'source': 'Server',
            **size_info
        })
        
        self._save_json()
    
    def _save_json(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_server_comparison(self, round_num, pt_results, onnx_results, size_info):
        """서버 비교 결과 출력"""
        speedup = pt_results['inference_time'] / onnx_results['inference_time'] if onnx_results['inference_time'] > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"라운드 {round_num} - 서버 글로벌 평가 (PyTorch vs ONNX)")
        print(f"{'='*70}")
        
        print(f"\n📦 모델 크기:")
        print(f"  PyTorch (.pth):      {size_info['pytorch_size_kb']:.2f} KB")
        print(f"  ONNX (원본):         {size_info['onnx_original_size_kb']:.2f} KB")
        print(f"  ONNX (Simplified):   {size_info['onnx_simplified_size_kb']:.2f} KB")
        print(f"  PyTorch 대비 절감:   {size_info['reduction_from_pytorch_pct']:.1f}%")
        print(f"  ONNX 원본 대비 절감: {size_info['reduction_from_original_pct']:.1f}%")
        
        print(f"\n⚡ PyTorch:")
        print(f"  Accuracy: {pt_results['accuracy']:.2f}%")
        print(f"  Macro F1: {pt_results['macro_f1']:.4f}")
        print(f"  추론 시간: {pt_results['inference_time']:.3f}초")
        print(f"  메모리: {pt_results['memory_usage']:.2f}MB")
        
        print(f"\n⚡ ONNX:")
        print(f"  Accuracy: {onnx_results['accuracy']:.2f}%")
        print(f"  Macro F1: {onnx_results['macro_f1']:.4f}")
        print(f"  추론 시간: {onnx_results['inference_time']:.3f}초")
        print(f"  메모리: {onnx_results['memory_usage']:.2f}MB")
        
        print(f"\n📊 비교:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  정확도 차이: {abs(pt_results['accuracy'] - onnx_results['accuracy']):.4f}%")
        print(f"{'='*70}\n")
    
    def generate_comparison_summary(self):
        """최종 비교 요약 생성"""
        summary_file = f"{self.results_dir}/comparison_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ONNX vs PyTorch 종합 비교 결과\n")
            f.write("="*70 + "\n\n")
            
            # 서버 결과 요약
            if self.results['server_results']:
                f.write("📊 서버 글로벌 평가 결과\n")
                f.write("-"*50 + "\n")
                
                pt_times = [r['pytorch']['inference_time'] for r in self.results['server_results']]
                onnx_times = [r['onnx']['inference_time'] for r in self.results['server_results']]
                speedups = [r['speedup'] for r in self.results['server_results']]
                
                f.write(f"PyTorch 평균 추론 시간: {np.mean(pt_times):.3f}초\n")
                f.write(f"ONNX 평균 추론 시간: {np.mean(onnx_times):.3f}초\n")
                f.write(f"평균 Speedup: {np.mean(speedups):.2f}x\n")
                
                # 최종 라운드 결과
                final = self.results['server_results'][-1]
                f.write(f"\n최종 라운드 결과:\n")
                f.write(f"  PyTorch Accuracy: {final['pytorch']['accuracy']:.2f}%\n")
                f.write(f"  ONNX Accuracy: {final['onnx']['accuracy']:.2f}%\n")
                f.write(f"  정확도 차이: {final['accuracy_diff']:.4f}%\n\n")
            
            # 모델 크기 요약
            if self.results['model_sizes']:
                f.write("📦 모델 크기 비교\n")
                f.write("-"*50 + "\n")
                
                # 서버 모델 크기 (마지막 라운드)
                server_sizes = [s for s in self.results['model_sizes'] if s['source'] == 'Server']
                if server_sizes:
                    last_size = server_sizes[-1]
                    f.write(f"PyTorch: {last_size['pytorch_size_kb']:.2f} KB\n")
                    f.write(f"ONNX (원본): {last_size['onnx_original_size_kb']:.2f} KB\n")
                    f.write(f"ONNX (Simplified): {last_size['onnx_simplified_size_kb']:.2f} KB\n")
                    f.write(f"PyTorch 대비 절감: {last_size['reduction_from_pytorch_pct']:.1f}%\n")
                    f.write(f"Simplify 절감: {last_size['reduction_from_original_pct']:.1f}%\n\n")
            
            # 클라이언트 결과 요약
            if self.results['client_results']:
                f.write("📱 클라이언트별 결과\n")
                f.write("-"*50 + "\n")
                
                client_ids = set([r.get('client_id', 0) for r in self.results['client_results']])
                
                for client_id in sorted(client_ids):
                    client_data = [r for r in self.results['client_results'] if r.get('client_id') == client_id]
                    
                    if client_data:
                        avg_speedup = np.mean([r.get('speedup', 0) for r in client_data])
                        avg_acc_diff = np.mean([r.get('accuracy_diff', 0) for r in client_data])
                        
                        f.write(f"\nClient {client_id}:\n")
                        f.write(f"  평균 Speedup: {avg_speedup:.2f}x\n")
                        f.write(f"  평균 정확도 차이: {avg_acc_diff:.4f}%\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"📊 비교 요약 저장: {summary_file}")

# ------------------------------------------------------------------------
# Custom Strategy (서버 ONNX 평가 추가)
# ------------------------------------------------------------------------
class ComparisonStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, test_loader, logger, num_classes=9, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.test_loader = test_loader
        self.logger = logger
        self.num_classes = num_classes
        self.current_round = 0
        self.onnx_manager = ServerONNXManager(model, logger.results_dir)
    
    def aggregate_fit(self, server_round, results, failures):
        self.current_round = server_round
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
            
            print(f"\n🔍 라운드 {server_round} 서버 글로벌 평가 중...")
            
            # 1. PyTorch 평가
            print("⚡ PyTorch 추론 중...")
            pt_results = evaluate_pytorch(self.model, self.test_loader, self.num_classes)
            
            # 2. ONNX 변환 및 평가
            print("🔄 ONNX 변환 중...")
            onnx_path = self.onnx_manager.export_and_simplify(use_simplifier=False)  # macOS M1 호환
            self.onnx_manager.load_session(onnx_path)
            size_info = self.onnx_manager.get_size_comparison()
            
            print("⚡ ONNX 추론 중...")
            onnx_results = evaluate_onnx(self.onnx_manager, self.test_loader, self.num_classes)
            
            # 결과 기록
            self.logger.log_server_result(server_round, pt_results, onnx_results, size_info)
            self.logger.print_server_comparison(server_round, pt_results, onnx_results, size_info)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """클라이언트 평가 결과 수집 및 기록"""
        if not results:
            return None, {}
        
        # 각 클라이언트 결과 기록
        for client_proxy, evaluate_res in results:
            metrics = evaluate_res.metrics
            if metrics and 'client_id' in metrics:
                self.logger.log_client_result(server_round, metrics)
        
        return super().aggregate_evaluate(server_round, results, failures)

# ------------------------------------------------------------------------
# 메인
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='ONNX vs PyTorch 비교 서버')
    parser.add_argument('--server_address', type=str, default='192.168.45.100:8080')
    parser.add_argument('--num_rounds', type=int, default=20)
    parser.add_argument('--min_clients', type=int, default=3)
    parser.add_argument('--experiment_name', type=str, default='onnx_comparison')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ONNX vs PyTorch 비교 서버 (서버 측 ONNX 평가 포함)")
    print("="*70)
    print(f"  실험 이름: {args.experiment_name}")
    print(f"  라운드: {args.num_rounds}")
    print("="*70 + "\n")
    
    test_loader = load_global_test_data(args.batch_size)
    model = SimpleCNN(num_classes=9).to(device)
    logger = ComparisonLogger(args.experiment_name)
    
    initial_parameters = [val.cpu().numpy() for val in model.state_dict().values()]
    
    strategy = ComparisonStrategy(
        model=model,
        test_loader=test_loader,
        logger=logger,
        num_classes=9,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_metrics_aggregation_fn=None,
    )
    
    print(f"🚀 서버 시작: {args.server_address}\n")
    
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy
    )
    
    logger.generate_comparison_summary()
    
    print("\n" + "="*70)
    print("✅ 비교 실험 완료!")
    print(f"📁 결과: {logger.results_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()