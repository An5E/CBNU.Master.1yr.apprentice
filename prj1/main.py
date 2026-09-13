# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. 전역 하이퍼파라미터 및 환경 설정
# ==========================================


HYPERPARAMS = {
    "hidden_dims": [64, 32],      
    "learning_rate": 0.001,
    "epochs": 100,               
    "batch_size": 256,
    "dropout_rate": 0.1,
    "patience": 5                
}

OUTPUT_DTIME = pd.to_datetime('now').strftime('%Y%m%d%H%M%S')

OUTPUT_EXCEL_FILE = f"experiment_epoch_mse_results.{OUTPUT_DTIME}.xlsx"

# ==========================================
# 2. Early Stopping 제어 클래스 정의
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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
def run_model_session(X_train, y_train, X_val, y_val, X_test, y_test, scaler=None):
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
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=False)
    
    t_train_x, t_train_y = torch.tensor(X_train_arr, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    t_val_x, t_val_y = (torch.tensor(X_val_arr, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)) if X_val_arr is not None else (None, None)
    t_test_x, t_test_y = torch.tensor(X_test_arr, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

    input_features_dim = X_train_arr.shape[1]
    model = GenerationPredictor(input_dim=input_features_dim, 
                                hidden_dims=HYPERPARAMS["hidden_dims"],
                                dropout_rate=HYPERPARAMS["dropout_rate"])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
    
    early_stopping = EarlyStopping(patience=HYPERPARAMS["patience"])
    epoch_mse_history = []

    for epoch in range(HYPERPARAMS["epochs"]):
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
                monitor_loss = val_mse   
            else:
                val_mse = None
                monitor_loss = test_mse  

        epoch_record = {
            "Epoch": epoch + 1,
            "Train_MSE": round(train_mse, 4),
            "Val_MSE": round(val_mse, 4) if val_mse is not None else "N/A",
            "Test_MSE": round(test_mse, 4)
        }
        epoch_mse_history.append(epoch_record)
        
        early_stopping(monitor_loss, model)
        if early_stopping.early_stop:
            print(f"   ↪ [Early Stopping] 에폭 {epoch + 1}에서 조기 종료.")
            model.load_state_dict(early_stopping.best_model_state)
            break

    model.eval()
    with torch.no_grad():
        final_preds = model(t_test_x).numpy()
        
    final_mse = mean_squared_error(y_test, final_preds)
    final_mae = mean_absolute_error(y_test, final_preds)
    final_r2 = r2_score(y_test, final_preds)
    final_rmse = np.sqrt(final_mse)
    
    metrics = {"MSE": final_mse, "MAE": final_mae, "R2": final_r2, "RMSE": final_rmse}
    return metrics, epoch_mse_history

# ==========================================
# 6. 전체 구동 메인 제어문 (비교 시트 1개 + 개별 에폭 시트 4개)
# ==========================================
def main():
    df = pd.read_csv("./assets/dataset_datagokr_dongseo_cb20_24.csv")
    
    print("🤖 시계열 데이터 파이프라인 인프라 구축 중...")
    # date_range = pd.date_range(start="2020-01-01 00:00", periods=5000, freq="h")
    # dummy_data = {
    #     "capacity_mw": np.random.uniform(170, 180, 5000), "reg_dt": date_range,
    #     "temp": np.random.uniform(-10, 35, 5000), "precip_mm": np.random.uniform(0, 5, 5000),
    #     "relhumid": np.random.uniform(30, 90, 5000), "snow_mm": np.random.uniform(0, 0.5, 5000),
    #     "windspd": np.random.uniform(0.5, 6.0, 5000), "cumulus_10th": np.random.randint(0, 10, 5000),
    #     "cumulus_3rd": np.random.randint(0, 5, 5000), "sun_duration_hr": np.random.uniform(0, 1, 5000),
    #     "extra_rad": np.random.uniform(0, 400, 5000), "rad": np.random.uniform(0, 300, 5000),
    #     "gen_mwh": np.random.uniform(10, 150, 5000), "is_leapyr": np.random.choice(['Y', 'N'], 5000)
    # }
    # df = pd.DataFrame(dummy_data)
    
    X, y = preprocess_time_series(df)
    total_len = len(df)
    
    # 종합 요약 지표 리포트용 컨테이너
    summary_metrics_list = []
    
    # 각 시트별 데이터프레임을 보관할 딕셔너리
    sheet_data_dict = {}
        
    # ----------------------------------------------------
    # [테스트 1] Train:Test (8:2) / Scaler: None
    # ----------------------------------------------------
    split_82 = int(total_len * 0.8)
    X_train_82, X_test_82 = X.iloc[:split_82], X.iloc[split_82:]
    y_train_82, y_test_82 = y[:split_82], y[split_82:]
    
    print("\n📈 실험 1: Split 8-2 (Scaler: None) 학습 가동...")
    metrics_82, history_82 = run_model_session(X_train_82, y_train_82, None, None, X_test_82, y_test_82, scaler=None)
    
    summary_metrics_list.append({
        "Data_Split": "8:2", "Scaler": "None", "Hidden_Dims": str(HYPERPARAMS["hidden_dims"]),
        "LR": HYPERPARAMS["learning_rate"], "MSE": metrics_82["MSE"], "MAE": metrics_82["MAE"], 
        "R2": metrics_82["R2"], "RMSE": metrics_82["RMSE"]
    })
    sheet_data_dict["Split_8-2_No_Scaler"] = pd.DataFrame(history_82)
    
    # ----------------------------------------------------
    # [테스트 2, 3, 4] Train:Val:Test (6:2:2) / Scaler 3종
    # ----------------------------------------------------
    split_60 = int(total_len * 0.6)
    split_80 = int(total_len * 0.8)
    
    X_train_622 = X.iloc[:split_60]
    X_val_622 = X.iloc[split_60:split_80]
    X_test_622 = X.iloc[split_80:]
    
    y_train_622 = y[:split_60]
    y_val_622 = y[split_60:split_80]
    y_test_622 = y[split_80:]
    
    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler()
    }
    
    for name, scaler in scalers.items():
        print(f"📊 실험 2: Split 6-2-2 (Scaler: {name}) 학습 가동...")
        metrics_622, history_622 = run_model_session(X_train_622, y_train_622, X_val_622, y_val_622, X_test_622, y_test_622, scaler=scaler)
        
        summary_metrics_list.append({
            "Data_Split": "6:2:2", "Scaler": name, "Hidden_Dims": str(HYPERPARAMS["hidden_dims"]),
            "LR": HYPERPARAMS["learning_rate"], "MSE": metrics_622["MSE"], "MAE": metrics_622["MAE"], 
            "R2": metrics_622["R2"], "RMSE": metrics_622["RMSE"]
        })
        sheet_data_dict[f"Split_6-2-2_{name}"] = pd.DataFrame(history_622)

    # --------------------------------
    # ====================# 
    # [멀티 시트 저장부] 최종 엑셀 빌드 및 파일 쓰기 가동
    # # --------------------------------
    # ====================
    df_summary = pd.DataFrame(summary_metrics_list)
    writer = pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine='openpyxl')

    # 시트 1: 통합 테스트별 평가지표 비교 리포트 생성
    df_summary.to_excel(writer, sheet_name="Model_Comparison_Report", index=False)
    # 시트 2~5: 순차 루프 기반 에폭 정보 시트 적재
    for sheet_name, df_history in sheet_data_dict.items():
        df_history.to_excel(writer, sheet_name=sheet_name, index=False)
        
    writer.close()
    print(f"\n🎉 [컴파일 성공] 요약 리포트 1개 + 개별 에폭 추이 4개 통합 엑셀 저장 완료 -> 파일명: {OUTPUT_EXCEL_FILE}")

if __name__ == "__main__":
    main()
