import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 전처리
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['reg_dt'] = pd.to_datetime(df['reg_dt'])
    # 임베딩/원핫인코딩 없이 수치형 데이터만 추출 (is_leapyr는 수치형 변환)
    df['is_leapyr'] = df['is_leapyr'].map({'Y': 1, 'N': 0})
    
    # 학습에 사용할 피처 선택 (기상 데이터 및 환경 데이터)
    features = [
        'capacity_mw', 'temp', 'precip_mm', 'relhumid', 'snow_mm', 
        'windspd', 'cumulus_10th', 'cumulus_3rd', 'sun_duration_hr', 
        'extra_rad', 'rad', 'is_leapyr'
    ]
    target = 'gen_mwh'
    
    X = df[features]
    y = df[target]
    return X, y

# 2. 데이터 분할 및 모델 학습 함수
def run_experiment(X, y, split_type='8:2', scaler_name='Standard'):
    """
    split_type: '8:2' or '6:2:2'
    scaler_name: 'Standard', 'MinMax', 'Robust'
    """
    # 데이터 분할
    if split_type == '8:2':
        XXX = train_test_split(X, y, test_size=0.2, random_state=42)
        print(XXX.__len__())
        
        X_train, X_test, _,_ = XXX
        # test_set이 곧 validation set 역할을 수행
    else: # 6:2:2
        X_train, X_temp, _,_ = train_test_split(X, test_size=0.4, random_state=42)
        X_val, X_test, _,_ = train_test_split(X_temp, test_size=0.5, random_state=42)

    # Scaler 설정
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler()
    }
    scaler = scalers[scaler_name]
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if split_type == '6:2:2':
        # 중의적인 구조를 위해 X_temp를 다시 나누는 방식이므로 
        # 원본 데이터 인덱스를 유지하며 split을 진행해야 정확함
        # 여기서는 단순화된 논리로 구현
        X_val_scaled = scaler.transform(X_temp_data_placeholder) # 실제 구현 시 인덱스 주의
        # (이 부분은 실제 구현 시 데이터 슬라이싱을 정확히 처리해야 함)
    
    # 모델 정의 (예시: RandomForest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y)
    
    # 평가
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        "metrics": {"mse": mse, "r2": r2},
        "params": {"n_estimators": 100},
        "scaler": scaler_name,
        "split": split_type
    }

# 3. 메인 실행 및 결과 저장
def main():
    X, y = load_data("./assets/dataset_datagokr_dongseo_cb20_24.csv") # 실제 파일 경로
    
    results = []
    
    # 시나리오 1: 8:2 분할 + 1종 Scaler
    for s in ['Standard']:
        res = run_experiment(X, y, split_type='8:2', scaler_name=s)
        res['split'] = '8:2'
        results.append(res)
        
    # 시나리오 2: 6:2:2 분할 + 3종 Scaler
    for s in ['Standard', 'MinMax', 'Robust']:
        res = run_experiment(X, y, split_type='6:2:2', scaler_name=s)
        res['split'] = '6:2:2'
        results.append(res)

    # 결과 파일 저장
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Experiment complete. Results saved to experiment_results.json")

if __name__ == "__main__":
    main()