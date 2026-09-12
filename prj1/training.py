import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

# 1. 데이터 로드 및 전처리
def prepare_data(df):
    # 날짜 컬럼 변환
    df['reg_dt'] = pd.to_datetime(df['reg_dt'])
    
    # 임베딩/원핫인코딩 없이 시간 정보 추출 (수치형 변환)
    df['hour'] = df['reg_dt'].dt.hour
    df['day_of_month'] = df['reg_dt'].dt.day
    df['month'] = df['reg_dt'].dt.month
    df['day_of_week'] = df['reg_dt'].dt.day_of_week
    df['is_leapyr'] = df['is_leapyr'].map({'Y': 1, 'N': 0})
    
    # 불필요한 원본 날짜 컬럼 제거
    df = df.drop(columns=['reg_dt'])
    return df

# 데이터 로드
raw_data = pd.read_csv("./assets/dataset_datagokr_dongseo_cb20_24.csv")
df = prepare_data(raw_data)

# 2. 실험 설정 및 모델 학습 클래스
class RegressionModelTrainer:
    def __init__(self, target_col='gen_mwh'):
        self.target_col = target_col
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        # 여러 Scaler를 비교하기 위해 딕셔너리로 관리
        self.scalers = {
            "Standard": StandardScaler(),
            "Min-Max": MinMaxScaler(),
            "Robust": RobustScaler()
        }

    def train_and_eval(self, df, mode='2_way'):
        # 특징과 타겟 분리
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # [수정 포인트] 인덱스 기반으로 정확하게 6:2:2 분할 (셔플 없이)
        n = len(df)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        if mode == '2_way':
            # 80/20 분할
            split_idx = int(n * 0.8)
            X_train_raw = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_test_raw = X.iloc[split_idx:]
            y_test = y.iloc[split_idx:]
            
            X_train, X_test = X_train_raw, X_test_raw
            y_train, y_test = y_train, y_test
            print(f"--- [Mode: Train/Test] ---")
        else:
            # 60/20/20 분할
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_val = X.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]
            X_test = X.iloc[val_end:]
            y_test = y.iloc[val_end:]
            print(f"--- [Mode: Train/Val/Test] ---")

        # Scaler별 결과 비교 루프
        for name, scaler in self.scalers.items():
            # 각 Scaler 학습 및 적용 (Train 데이터로만 fit)
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # 모델 학습
            self.model.fit(X_train_scaled, y_train)
            
            # 결과 계산
            val_r2 = self.model.score(X_val_scaled, y_val)
            test_r2 = self.model.score(X_test_scaled, y_test)
            test_mae = mean_absolute_error(y_test, self.model.predict(X_test_scaled))
            
            print(f"[{name}] Val R2: {val_r2:.4f} | Test R2: {test_r2:.4f} | Test MAE: {test_mae:.4f}")


# 실행 예시
df_processed = df # 이미 밖에서 로드된 데이터 사용
trainer = RegressionModelTrainer()
# 6:2:2 비율을 테스트하려면 mode='3_way'를 사용하세요.
trainer.train_and_eval(df_processed, mode='3_way')