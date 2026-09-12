import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 1. 데이터 전처리 함수
def preprocess_data(df):
    df = df.copy()
    # 날짜 처리
    df['reg_dt'] = pd.to_datetime(df['reg_dt'])
    df['hour'] = df['reg_dt'].dt.hour
    df['day_of_month'] = df['reg_dt'].dt.day
    df['month'] = df['reg_dt'].dt.month
    df['day_of_week'] = df['reg_dt'].dt.dayofweek
    df['is_leapyr'] = df['is_leapyr'].map({'Y': 1, 'N': 0})
    
    # 불필요한 컬럼 제거 및 타겟 설정
    features = ['capacity_mw', 'temp', 'precip_mm', 'relhumid', 'snow_mm', 
                 'windspd', 'cumulus_10th', 'cumulus_3rd', 'sun_duration_hr', 
                 'extra_rad', 'rad', 'hour', 'day_of_month', 'month', 
                 'day_of_week', 'is_leapyr']
    target = 'gen_mwh'
    
    X = df[features]
    y = df[target]
    return X, y

# 2. 실험 실행 클래스
class EvaluationPipeline:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.scalers = {
            "Standard": StandardScaler(),
            "Min-Max": MinMaxScaler(),
            "Robust": RobustScaler()
        }

    def run_experiment(self):
        # [Step 1] 데이터 분할 (6:2:2)
        # 먼저 80% (Train+Val)와 20% (Test)를 나눔
        X_temp, X_test = train_test_split(self.X, test_size=0.2, shuffle=False)
        # 그 다음 80% 중 25%를 Val로 할당 (전체의 20%가 됨)
        X_train, X_val = train_test_split(X_temp, test_size=0.25, shuffle=False)
        
        # 타겟 데이터도 비율에 맞춰 분할 (인덱스 유지)
        # (실제 구현 시 인덱스를 추적하거나 동일한 random_state/shuffle 설정을 사용해야 함)
        # 여기서는 편의상 단순 분할을 위해 전체를 한번에 처리하는 논리 적용
        
        # 인덱스 기반 정확한 분할 (셔플 없이)
        indices = np.arange(len(self.X))
        train_idx = indices[:int(len(indices)*0.6)]
        val_idx = indices[int(len(indices)*0.6):int(len(indices)*0.8)]
        test_idx = indices[int(len(indices)*0.8):]
        
        X_train, X_val, X_test = self.X.iloc[train_idx], self.X.iloc[val_idx], self.X.iloc[test_idx]
        y_train, y_val, y_test = self.y.iloc[train_idx], self.y.iloc[val_idx], self.y.iloc[test_idx]

        results = {}

        # [Step 2] Scaler별 실험 루프
        for name, scaler in self.scalers.items():
            # 각 Scaler 학습 및 적용
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # 모델 정의 및 학습 (RandomForest 또는 다른 회귀 모델)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # 평가
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)
            
            results[name] = {
                "Val R2": r2_score(y_val, val_pred),
                "Test R2": r2_score(y_test, test_pred),
                "Test MAE": mean_absolute_error(y_test, test_pred)
            }
            
            print(f"[{name}] Val R2: {results[name]['Val R2']:.4f} | Test R2: {results[name]['Test R2']:.4f}")

        return results

# 3. 실행부
df = pd.read_csv('your_data_path.csv')
X, y = preprocess_data(df)
pipeline = EvaluationPipeline(X, y)
results = pipeline.run_experiment()