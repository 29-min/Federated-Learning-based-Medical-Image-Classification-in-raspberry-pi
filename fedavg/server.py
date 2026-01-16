"""
개선된 연합학습 서버 - 글로벌 평가 기능 추가
목표: PathMNIST 전체 테스트셋으로 글로벌 모델 성능 평가
"""

import flwr as fl
from flwr.server.strategy import FedAvg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import argparse
import os
import medmnist
from medmnist.info import INFO
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import json
from datetime import datetime


class SimpleCNN(nn.Module):
    """SimpleCNN - PathMNIST용 (9개 클래스)"""
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


def load_global_test_data(batch_size=128):
    """
    글로벌 테스트 데이터셋 로드
    - 모든 클래스가 포함된 전체 테스트셋
    - 클라이언트에게 분할되지 않음
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # 전체 테스트 데이터 (분할하지 않음!)
    test_dataset = DataClass(split='test', transform=transform, download=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n{'='*70}")
    print(f"🌍 글로벌 테스트 데이터셋 로드 완료")
    print(f"{'='*70}")
    print(f"  - 총 테스트 샘플 수: {len(test_dataset)}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 클래스 수: {len(info['label'])}")
    print(f"  ⚠️  이 데이터는 클라이언트들에게 분할되지 않은 전체 테스트셋입니다.")
    print(f"  ✅ 모델의 진짜 일반화 성능을 측정할 수 있습니다!")
    print(f"{'='*70}\n")
    
    return test_loader, len(info['label'])


def get_parameters(model):
    """모델 파라미터를 리스트로 변환"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """파라미터를 모델에 적용"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def evaluate_global_model(model, test_loader, device, round_num, num_classes=9):
    """
    글로벌 모델을 전체 테스트셋으로 평가
    - 전체 정확도, F1 score
    - 클래스별 정확도
    - Confusion matrix
    """
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).squeeze().long()
            output = model(data)
            
            test_loss += criterion(output, target).item()
            
            # Softmax probabilities
            probs = torch.softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 메트릭 계산
    test_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    f1_macro = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
    
    # 클래스별 정확도
    conf_matrix = confusion_matrix(all_targets, all_predictions, labels=list(range(num_classes)))
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"🌍 ROUND {round_num} - GLOBAL MODEL EVALUATION (Unbiased Test Set)")
    print(f"{'='*70}")
    print(f"  📊 Overall Performance:")
    print(f"     - Test Loss:           {test_loss:.4f}")
    print(f"     - Test Accuracy:       {accuracy:.2f}%")
    print(f"     - F1 Score (Macro):    {f1_macro:.4f}")
    print(f"     - F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"\n  📈 Per-Class Accuracy:")
    for i in range(num_classes):
        bar_length = int(class_accuracy[i] * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"     Class {i}: {class_accuracy[i]*100:>5.1f}%  {bar}")
    print(f"{'='*70}\n")
    
    results = {
        'round': round_num,
        'loss': test_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'class_accuracy': class_accuracy.tolist(),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return results


def save_server_checkpoint(model, round_num, checkpoint_dir="checkpoints/server"):
    """서버 체크포인트 저장"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/round_{round_num}.pt"
    torch.save({
        'round': round_num,
        'model_state_dict': model.state_dict(),
    }, checkpoint_path)
    print(f"💾 서버 체크포인트 저장: {checkpoint_path}")


def load_server_checkpoint(model, checkpoint_dir="checkpoints/server"):
    """서버 체크포인트 로드"""
    import glob
    
    checkpoints = glob.glob(f"{checkpoint_dir}/round_*.pt")
    
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        round_num = checkpoint['round']
        
        print(f"✅ 서버 체크포인트 로드: {latest_checkpoint}")
        print(f"   라운드 {round_num}부터 재개합니다.")
        return round_num
    else:
        print(f"🆕 새로운 서버 학습 시작")
        return 0


class GlobalEvaluationStrategy(FedAvg):
    """
    글로벌 평가 기능이 추가된 FedAvg 전략
    - 매 라운드마다 글로벌 테스트셋으로 평가
    - 결과를 CSV/JSON으로 저장
    """
    
    def __init__(
        self, 
        model, 
        global_test_loader,
        num_classes=9,
        checkpoint_dir="checkpoints/server",
        results_dir="results",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.global_test_loader = global_test_loader
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_results = []
        
        # 결과 저장 디렉토리 생성
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n🔧 GlobalEvaluationStrategy 초기화")
        print(f"   - Device: {self.device}")
        print(f"   - Checkpoint Dir: {checkpoint_dir}")
        print(f"   - Results Dir: {results_dir}")
    
    def aggregate_fit(self, server_round, results, failures):
        """
        FedAvg 집계 후:
        1. 체크포인트 저장
        2. 글로벌 테스트셋으로 평가
        3. 결과 저장
        """
        # FedAvg 기본 집계
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # 1. 집계된 파라미터를 모델에 적용
            parameters_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            set_parameters(self.model, parameters_ndarrays)
            
            # 2. 체크포인트 저장
            save_server_checkpoint(self.model, server_round, self.checkpoint_dir)
            
            # 3. 글로벌 테스트셋으로 평가
            eval_results = evaluate_global_model(
                self.model, 
                self.global_test_loader, 
                self.device, 
                server_round,
                self.num_classes
            )
            
            # 4. 결과 저장
            self.global_results.append(eval_results)
            self.save_results()
            
            # 5. 클라이언트 로컬 성능도 함께 저장
            self.save_client_metrics(server_round, results)
        
        return aggregated_parameters, aggregated_metrics
    
    def save_results(self):
        """글로벌 평가 결과를 CSV와 JSON으로 저장"""
        # CSV로 저장 (시각화 용이)
        df_data = []
        for result in self.global_results:
            row = {
                'round': result['round'],
                'loss': result['loss'],
                'accuracy': result['accuracy'],
                'f1_macro': result['f1_macro'],
                'f1_weighted': result['f1_weighted']
            }
            # 클래스별 정확도 추가
            for i, acc in enumerate(result['class_accuracy']):
                row[f'class_{i}_acc'] = acc * 100
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = f"{self.results_dir}/global_evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        
        # JSON으로 저장 (전체 정보)
        json_path = f"{self.results_dir}/global_evaluation_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.global_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 글로벌 평가 결과 저장:")
        print(f"   - CSV: {csv_path}")
        print(f"   - JSON: {json_path}")
    
    def save_client_metrics(self, server_round, results):
        """클라이언트 로컬 평가 결과 저장"""
        client_metrics = []
        
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            client_metrics.append({
                'round': server_round,
                'client_id': client_proxy.cid,
                'num_examples': fit_res.num_examples,
                'metrics': metrics
            })
        
        # 라운드별로 저장
        json_path = f"{self.results_dir}/client_metrics_round_{server_round}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(client_metrics, f, indent=2, ensure_ascii=False)


def print_final_summary(results_dir="results"):
    """최종 결과 요약 출력"""
    csv_path = f"{results_dir}/global_evaluation_results.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        print(f"\n{'='*70}")
        print(f"📊 최종 결과 요약")
        print(f"{'='*70}")
        print(f"\n  전체 라운드 통계:")
        print(f"     - 최초 정확도 (Round 1):  {df['accuracy'].iloc[0]:.2f}%")
        print(f"     - 최종 정확도 (Round {len(df)}): {df['accuracy'].iloc[-1]:.2f}%")
        print(f"     - 정확도 향상:            +{df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]:.2f}%p")
        print(f"     - 평균 정확도:            {df['accuracy'].mean():.2f}%")
        print(f"     - 최고 정확도:            {df['accuracy'].max():.2f}% (Round {df['accuracy'].idxmax() + 1})")
        
        print(f"\n  F1 Score 통계:")
        print(f"     - 최종 F1 (Macro):        {df['f1_macro'].iloc[-1]:.4f}")
        print(f"     - 최종 F1 (Weighted):     {df['f1_weighted'].iloc[-1]:.4f}")
        
        print(f"\n  클래스별 최종 정확도:")
        for i in range(9):
            col_name = f'class_{i}_acc'
            if col_name in df.columns:
                final_acc = df[col_name].iloc[-1]
                print(f"     Class {i}: {final_acc:.1f}%")
        
        print(f"{'='*70}\n")


def main():
    """서버 메인 함수"""
    parser = argparse.ArgumentParser(description='개선된 연합학습 서버 (글로벌 평가)')
    parser.add_argument('--num_rounds', type=int, default=20, help='연합학습 라운드 수')
    parser.add_argument('--num_clients', type=int, default=3, help='전체 클라이언트 수')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha (참고용)')
    parser.add_argument('--port', type=int, default=8080, help='서버 포트')
    parser.add_argument('--batch_size', type=int, default=128, help='글로벌 테스트 배치 크기')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🌍 개선된 연합학습 서버 - 글로벌 평가 기능")
    print("="*70)
    
    # 실험 설정
    config = {
        'num_rounds': args.num_rounds,
        'num_clients': args.num_clients,
        'fraction_fit': 1.0,
        'min_fit_clients': args.num_clients,
        'min_available_clients': args.num_clients,
        'dataset': 'PathMNIST',
        'num_classes': 9,
        'model': 'SimpleCNN',
        'data_distribution': f'Non-IID (Dirichlet alpha={args.alpha})',
        'strategy': 'FedAvg + Global Evaluation'
    }
    
    print(f"\n⚙️  서버 설정:")
    print(f"  - 모델: {config['model']}")
    print(f"  - 데이터셋: {config['dataset']} ({config['num_classes']}개 클래스)")
    print(f"  - 데이터 분포: {config['data_distribution']}")
    print(f"  - 전략: {config['strategy']}")
    print(f"  - 라운드 수: {config['num_rounds']}")
    print(f"  - 클라이언트 수: {config['num_clients']}")
    print(f"  - 참여 비율: {config['fraction_fit']*100:.0f}%")
    
    # 글로벌 테스트 데이터 로드
    global_test_loader, num_classes = load_global_test_data(batch_size=args.batch_size)
    
    # 초기 모델 생성
    model = SimpleCNN(num_classes=num_classes)
    print(f"\n🔧 SimpleCNN 모델 생성")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   총 파라미터 수: {total_params:,}")
    
    # 체크포인트 로드 시도
    start_round = load_server_checkpoint(model)
    
    initial_parameters = get_parameters(model)
    parameters = fl.common.ndarrays_to_parameters(initial_parameters)
    
    # 글로벌 평가 전략 생성
    strategy = GlobalEvaluationStrategy(
        model=model,
        global_test_loader=global_test_loader,
        num_classes=num_classes,
        checkpoint_dir="checkpoints/server",
        results_dir="results",
        fraction_fit=config['fraction_fit'],
        fraction_evaluate=1.0,
        min_fit_clients=config['min_fit_clients'],
        min_evaluate_clients=config['num_clients'],
        min_available_clients=config['min_available_clients'],
        initial_parameters=parameters
    )
    
    print(f"\n{'='*70}")
    print(f"🚀 서버 시작")
    print(f"{'='*70}")
    print(f"  - 주소: 0.0.0.0:{args.port}")
    print(f"  - 대기 중: {config['num_clients']}개 클라이언트 연결 대기")
    
    print(f"\n💡 클라이언트 실행 명령어 (터미널 {config['num_clients']}개 필요):")
    print(f"{'='*70}")
    for i in range(config['num_clients']):
        print(f"  터미널 {i+1}:")
        print(f"  python3 client.py --client_id {i} --total_clients {config['num_clients']} --alpha {args.alpha}")
        print()
    
    print(f"{'='*70}")
    print(f"⏳ 클라이언트 연결 대기 중...")
    print(f"{'='*70}\n")
    
    # 서버 시작
    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
            strategy=strategy
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  서버 중단됨")
    finally:
        # 최종 결과 요약
        print_final_summary()
        
        print(f"\n{'='*70}")
        print(f"✅ 연합학습 완료!")
        print(f"{'='*70}")
        print(f"\n📁 저장된 파일:")
        print(f"  - 체크포인트: checkpoints/server/")
        print(f"  - 글로벌 평가 결과: results/global_evaluation_results.csv")
        print(f"  - 상세 결과: results/global_evaluation_results.json")
        print(f"  - 클라이언트 메트릭: results/client_metrics_round_*.json")
        
        print(f"\n💡 다음 단계:")
        print(f"  1. results/global_evaluation_results.csv로 시각화")
        print(f"  2. 로컬 평가 vs 글로벌 평가 비교")
        print(f"  3. Baseline (중앙집중형) 실험과 비교")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()