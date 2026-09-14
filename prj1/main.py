# -*- coding: utf-8 -*-
import os
import json
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. 자동 탐색(Search) 하이퍼파라미터 후보군 정의
# ==========================================
TUNING_GRID = {
    "hidden_dims": [[64, 32],
                    # [32, 16]
    ],  # 탐색할 은닉층 구조 후보
    "learning_rate": [
        # 0.001, 
        0.005
    ],             # 탐색할 학습률 후보
    "batch_size":[256],
    "dropout_rate": [
        0.1,
        # 0.2
    ],
    "max_epochs":[100],
    "patience": [5]                       # 연속 5회 기준 손실 미개선 시 조기 종료
}

OUTPUT_DT = pd.to_datetime('now').strftime('%Y%m%d%H%M%S')
OUTPUT_EXCEL_FILE = f"alt_experiment_5_cases_results.{OUTPUT_DT}.xlsx"

# ==========================================
# 2. Early Stopping 제어 클래스 정의
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, monitor_loss, epoch, model):
        if monitor_loss < self.best_loss - self.min_delta:
            self.best_loss = monitor_loss
            self.best_epoch = epoch
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==========================================
# 3. PyTorch 회귀 MLP 모델 아키텍처 정의
# ==========================================
class GenerationPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1):
        super(GenerationPredictor, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))  
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. 데이터 전처리 및 주기적 함수 변환 (Sin/Cos)
# ==========================================
def preprocess_time_series(df):
    df = df.copy()
    df['reg_dt'] = pd.to_datetime(df['reg_dt'])
    df = df.sort_values('reg_dt').reset_index(drop=True)
    
    month = df['reg_dt'].dt.month
    hour = df['reg_dt'].dt.hour
    
    df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    
    df['is_leapyr'] = df['is_leapyr'].map({'Y': 1.0, 'N': 0.0}).fillna(0.0)
    df = df.drop(columns=['reg_dt'])
    
    X = df.drop(columns=['gen_mwh']).astype(np.float32)
    y = df['gen_mwh'].values.astype(np.float32).reshape(-1, 1)
    
    return X, y

# ==========================================
# 5. 세션 학습 엔진 및 에폭별 MSE 기록 인터페이스
# ==========================================
def run_model_session(X_train, y_train, X_val, y_val, X_test, y_test, config, scaler=None):
    if scaler is not None:
        X_train_arr = scaler.fit_transform(X_train)
        X_val_arr = scaler.transform(X_val) if X_val is not None else None
        X_test_arr = scaler.transform(X_test)
    else:
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_val_arr = X_val.values if (X_val is not None and isinstance(X_val, pd.DataFrame)) else X_val
        X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    train_dataset = TensorDataset(torch.tensor(X_train_arr, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    
    t_train_x, t_train_y = torch.tensor(X_train_arr, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    t_val_x, t_val_y = (torch.tensor(X_val_arr, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)) if X_val_arr is not None else (None, None)
    t_test_x, t_test_y = torch.tensor(X_test_arr, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    input_features_dim = X_train_arr.shape[1]
    model = GenerationPredictor(input_dim=input_features_dim, 
                                hidden_dims=config["hidden_dims"],
                                dropout_rate=config["dropout_rate"])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    early_stopping = EarlyStopping(patience=config["patience"])
    epoch_mse_history = []
    last_epoch = config["max_epochs"]

    for epoch in range(config["max_epochs"]):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            train_mse = criterion(model(t_train_x), t_train_y).item()
            test_mse = criterion(model(t_test_x), t_test_y).item()
            
            if t_val_x is not None:
                val_mse = criterion(model(t_val_x), t_val_y).item()
                monitor_loss = val_mse   # 6:2:2 구조는 Val MSE 기준으로 조기종료 및 베스트 추적
            else:
                val_mse = None
                monitor_loss = test_mse  # 8:2 구조는 별도 검증셋이 없으므로 Test MSE 기준으로 추적

        epoch_record = {
            "Epoch": epoch + 1,
            "Train_MSE": round(train_mse, 4),
            "Val_MSE": round(val_mse, 4) if val_mse is not None else "N/A",
            "Test_MSE": round(test_mse, 4)
        }
        epoch_mse_history.append(epoch_record)
        
        early_stopping(monitor_loss, epoch + 1, model)
        if early_stopping.early_stop:
            last_epoch = epoch + 1
            model.load_state_dict(early_stopping.best_model_state)
            break

    # 최적 시점(Best Epoch) 가중치 상태에서 최종 성능 복원 및 평가
    model.eval()
    with torch.no_grad():
        final_preds = model(t_test_x).numpy()
        
    final_mse = mean_squared_error(y_test, final_preds)
    final_mae = mean_absolute_error(y_test, final_preds)
    final_r2 = r2_score(y_test, final_preds)
    final_rmse = np.sqrt(final_mse)
    
    metrics = {
        "MSE": final_mse, "MAE": final_mae, "R2": final_r2, "RMSE": final_rmse,
        "Best_MSE": final_mse,  # 베스트 에폭 가중치 시점의 최종 Test 데이터 기준 MSE
        "Best_Epoch": early_stopping.best_epoch,
        "Last_Epoch": last_epoch
    }
    return metrics, epoch_mse_history

# ==========================================
# 6. 메인 제어문 (수정 조건 반영 5대 케이스 실험 가동)
# ==========================================
def main():
    df = pd.read_csv("./assets/dataset_datagokr_dongseo_cb20_24.csv")
    
    print("🤖 시계열 데이터 파이프라인 인프라 구축 중...")
    
    X, y = preprocess_time_series(df)
    total_len = len(df)
    
    summary_metrics_list = []
    sheet_data_dict = {}

    keys, values = zip(*TUNING_GRID.items())
    experiments_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # ----------------------------------------------------
    # 데이터셋 구조 분할 선행 정의 (순서 보존 슬라이싱)
    # ----------------------------------------------------
    # [8:2] 데이터셋 분할
    split_82 = int(total_len * 0.8)
    X_train_82, X_test_82 = X.iloc[:split_82], X.iloc[split_82:]
    y_train_82, y_test_82 = y[:split_82], y[split_82:]
    
    # [6:2:2] 데이터셋 분할
    split_60 = int(total_len * 0.6)
    split_80 = int(total_len * 0.8)
    X_train_622 = X.iloc[:split_60]
    X_val_622 = X.iloc[split_60:split_80]
    X_test_622 = X.iloc[split_80:]
    y_train_622 = y[:split_60]
    y_val_622 = y[split_60:split_80]
    y_test_622 = y[split_80:]

    # 하이퍼파라미터 루프 실행
    for idx, config in enumerate(experiments_configs):
        run_id = idx + 1
        print(f"\n🚀 [조합 실험 {run_id}/{len(experiments_configs)}] 파라미터 조합 검증 개시...")
        
        # 테스트#1) Train:Test (8:2) 및 Scaler 미적용
        m1, h1 = run_model_session(X_train_82, y_train_82, None, None, X_test_82, y_test_82, config, scaler=None)
        summary_metrics_list.append({
            "Run_ID": f"Run_{run_id}", "Test_No": "테스트#1", "Data_Split": "8:2", "Scaler": "None", 
            "Hidden_Dims": str(config["hidden_dims"]), "LR": config["learning_rate"], "Dropout": config["dropout_rate"],
            "MSE": m1["MSE"], "MAE": m1["MAE"], "R2": m1["R2"], "RMSE": m1["RMSE"],
            "Best_MSE(Test)": m1["Best_MSE"], "Best_Epoch": m1["Best_Epoch"], "Last_Epoch": m1["Last_Epoch"]
        })
        sheet_data_dict[f"Run{run_id}_T1_82_None"] = pd.DataFrame(h1)
        
        # 테스트#2) Train:Val:Test (6:2:2) 및 Scaler 미적용
        m2, h2 = run_model_session(X_train_622, y_train_622, X_val_622, y_val_622, X_test_622, y_test_622, config, scaler=None)
        summary_metrics_list.append({"Run_ID": f"Run_{run_id}", "Test_No": "테스트#2", "Data_Split": "6:2:2", "Scaler": "None","Hidden_Dims": str(config["hidden_dims"]), 
        "LR": config["learning_rate"], "Dropout": config["dropout_rate"], "MSE": m2["MSE"], "MAE": m2["MAE"], "R2": m2["R2"], 
        "RMSE": m2["RMSE"],"Best_MSE(Test)": m2["Best_MSE"], "Best_Epoch": m2["Best_Epoch"], "Last_Epoch": m2["Last_Epoch"]})
        sheet_data_dict[f"Run{run_id}_T2_622_None"] = pd.DataFrame(h2)

        # 6:2:2 기반의 스케일러 매핑 딕셔너리 구성 (테스트 #3, #4, #5)
        scalers_82 = {"테스트#3'_StandardScaler": StandardScaler(),"테스트#4_MinMaxScaler": MinMaxScaler(),"테스트#5_RobustScaler": RobustScaler()}
        scalers_622 = {"테스트#6_StandardScaler": StandardScaler(),"테스트#7_MinMaxScaler": MinMaxScaler(),"테스트#8_RobustScaler": RobustScaler()}

        for case_name, scaler in scalers_82.items():
            test_no, scaler_name = case_name.split("_")
            m_case, h_case = run_model_session(X_train_82, y_train_82, None, None, X_test_82, y_test_82, config, scaler=scaler)
            summary_metrics_list.append({"Run_ID": f"Run_{run_id}", "Test_No": test_no, "Data_Split": "8:2", "Scaler": scaler_name,"Hidden_Dims": str(config["hidden_dims"]), 
                "LR": config["learning_rate"], "Dropout": config["dropout_rate"] ,"MSE": m_case["MSE"], "MAE": m_case["MAE"], "R2": m_case["R2"], "RMSE": m_case["RMSE"],"Best_MSE(Test)": m_case["Best_MSE"], 
            "Best_Epoch": m_case["Best_Epoch"], "Last_Epoch": m_case["Last_Epoch"]})
            sheet_data_dict[f"Run{run_id}{test_no}{scaler_name}"] = pd.DataFrame(h_case)

        for case_name, scaler in scalers_622.items():
            test_no, scaler_name = case_name.split("_")
            m_case, h_case = run_model_session(X_train_622, y_train_622, X_val_622, y_val_622, X_test_622, y_test_622, config, scaler=scaler)
            summary_metrics_list.append({"Run_ID": f"Run_{run_id}", "Test_No": test_no, "Data_Split": "6:2:2", "Scaler": scaler_name,"Hidden_Dims": str(config["hidden_dims"]), 
                "LR": config["learning_rate"], "Dropout": config["dropout_rate"] ,"MSE": m_case["MSE"], "MAE": m_case["MAE"], "R2": m_case["R2"], "RMSE": m_case["RMSE"],"Best_MSE(Test)": m_case["Best_MSE"], 
            "Best_Epoch": m_case["Best_Epoch"], "Last_Epoch": m_case["Last_Epoch"]})
            sheet_data_dict[f"Run{run_id}{test_no}{scaler_name}"] = pd.DataFrame(h_case)

        # ----------------------------------------------------
        # 최종 멀티 시트 엑셀 컴파일
        # ----------------------------------------------------
        df_summary = pd.DataFrame(summary_metrics_list)
        writer = pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl')
        # 1번 시트: 총 5개 테스트 케이스 총괄 결과 비교표 저장
        df_summary.to_excel(writer, sheet_name="Tuning_Comparison_Report", index=False)
        # 나머지 시트: 개별 테스트의 에폭 경과 로그 파일 세팅
        for sheet_name, df_history in sheet_data_dict.items():
            safe_sheet_name = sheet_name.replace("#", "")[:30] 

            # 특수문자 완화 및 31자 제한 우회
            df_history.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        writer.close()

    print(f"\n🎉 [완료] 수정된 5개 테스트 케이스 실험 완료. 결과 저장 파일: {OUTPUT_EXCEL_FILE}")

if __name__ == "__main__":
    main()