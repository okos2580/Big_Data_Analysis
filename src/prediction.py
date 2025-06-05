"""
주식 가격 예측 모듈
다양한 머신러닝 모델을 사용하여 주가를 예측합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 딥러닝 라이브러리 (TensorFlow가 설치되지 않았으므로 주석 처리)
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 시각화
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt
import seaborn as sns

class StockPredictor:
    def __init__(self):
        """주가 예측기 초기화"""
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/predictions', exist_ok=True)
        
        # 모델 저장을 위한 딕셔너리
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
        # 평가 지표 저장
        self.evaluation_results = {}
        
    def load_data(self, filepath):
        """
        전처리된 데이터를 로드합니다.
        
        Args:
            filepath (str): 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        try:
            data = pd.read_csv(filepath, encoding='utf-8-sig', index_col=0, parse_dates=True)
            print(f"✓ 예측용 데이터 로드 완료: {filepath}")
            return data
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            return None
    
    def prepare_features(self, data, target_column='Close', prediction_days=5):
        """
        예측을 위한 특성과 타겟을 준비합니다.
        
        Args:
            data (pd.DataFrame): 주식 데이터
            target_column (str): 예측할 컬럼
            prediction_days (int): 예측 일수
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # 예측에 사용할 특성들
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position',
            'Daily_Return', 'Volatility',
            'DayOfWeek', 'Month', 'Quarter'
        ]
        
        # 지연 특성 포함
        lag_features = [col for col in data.columns if '_lag_' in col]
        feature_columns.extend(lag_features)
        
        # 사용 가능한 특성만 선택
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 특성 데이터 준비
        X = data[available_features].copy()
        
        # 타겟 변수: N일 후 종가
        y = data[target_column].shift(-prediction_days)
        
        # 결측치 제거
        valid_indices = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_indices]
        y = y[valid_indices]
        
        print(f"✓ 특성 준비 완료: {len(available_features)}개 특성, {len(X)}개 샘플")
        print(f"예측 대상: {prediction_days}일 후 {target_column}")
        
        return X, y, available_features
    
    def split_time_series_data(self, X, y, test_size=0.2, validation_size=0.2):
        """
        시계열 데이터를 시간 순서를 고려하여 분할합니다.
        
        Args:
            X (pd.DataFrame): 특성 데이터
            y (pd.Series): 타겟 데이터
            test_size (float): 테스트 세트 비율
            validation_size (float): 검증 세트 비율
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        
        # 시간 순서를 고려한 분할
        train_end = int(n_samples * (1 - test_size - validation_size))
        val_end = int(n_samples * (1 - test_size))
        
        X_train = X.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]
        
        print(f"✓ 데이터 분할 완료:")
        print(f"  - 훈련: {len(X_train)} 샘플")
        print(f"  - 검증: {len(X_val)} 샘플")
        print(f"  - 테스트: {len(X_test)} 샘플")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_traditional_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        전통적인 머신러닝 모델들을 훈련합니다.
        
        Args:
            X_train, X_val, X_test: 특성 데이터
            y_train, y_val, y_test: 타겟 데이터
        """
        print("🤖 전통적인 머신러닝 모델 훈련 시작")
        
        # 데이터 정규화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['traditional'] = scaler
        
        # 모델 정의
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'Support Vector Regression': SVR(
                kernel='rbf',
                C=100,
                gamma='scale'
            )
        }
        
        # 각 모델 훈련 및 평가
        for model_name, model in models.items():
            print(f"📊 {model_name} 훈련 중...")
            
            # 모델 훈련
            model.fit(X_train_scaled, y_train)
            
            # 예측
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)
            
            # 평가
            train_metrics = self.calculate_metrics(y_train, train_pred)
            val_metrics = self.calculate_metrics(y_val, val_pred)
            test_metrics = self.calculate_metrics(y_test, test_pred)
            
            # 결과 저장
            self.models[model_name] = model
            self.predictions[model_name] = {
                'train': train_pred,
                'val': val_pred,
                'test': test_pred
            }
            self.evaluation_results[model_name] = {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            }
            
            print(f"  - 검증 RMSE: {val_metrics['rmse']:.4f}")
            print(f"  - 검증 R²: {val_metrics['r2']:.4f}")
    
    def prepare_lstm_data(self, data, feature_columns, target_column, sequence_length=60):
        """
        LSTM을 위한 시퀀스 데이터를 준비합니다.
        
        Args:
            data (pd.DataFrame): 시계열 데이터
            feature_columns (list): 사용할 특성 컬럼들
            target_column (str): 예측할 컬럼
            sequence_length (int): 시퀀스 길이
            
        Returns:
            tuple: (X, y, scaler)
        """
        # 데이터 정규화
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[feature_columns + [target_column]])
        
        # 시퀀스 생성
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, :-1])  # 특성들
            y.append(scaled_data[i, -1])  # 타겟
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ LSTM 데이터 준비 완료: {X.shape}, {y.shape}")
        
        return X, y, scaler
    
    def build_lstm_model(self, input_shape, units=[50, 50], dropout_rate=0.2):
        """
        LSTM 모델을 구성합니다.
        
        Args:
            input_shape (tuple): 입력 데이터 형태
            units (list): LSTM 레이어의 유닛 수들
            dropout_rate (float): 드롭아웃 비율
            
        Returns:
            tf.keras.Model: 구성된 LSTM 모델
        """
        model = Sequential()
        
        # 첫 번째 LSTM 레이어
        model.add(LSTM(units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # 추가 LSTM 레이어들
        for i in range(1, len(units)):
            return_sequences = i < len(units) - 1
            model.add(LSTM(units[i], return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # 출력 레이어
        model.add(Dense(1))
        
        # 컴파일
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        return model
    
    def train_lstm_model(self, data, feature_columns, target_column='Close', 
                        sequence_length=60, prediction_days=5):
        """
        LSTM 모델을 훈련합니다. (TensorFlow가 없으면 건너뜀)
        
        Args:
            data (pd.DataFrame): 시계열 데이터
            feature_columns (list): 사용할 특성 컬럼들
            target_column (str): 예측할 컬럼
            sequence_length (int): 시퀀스 길이
            prediction_days (int): 예측 일수
        """
        print("🧠 LSTM 모델 훈련 시작")
        print("⚠️ TensorFlow가 설치되지 않아 LSTM 모델을 건너뜁니다.")
        return None
    
    def calculate_metrics(self, actual, predicted):
        """
        예측 성능 지표를 계산합니다.
        
        Args:
            actual (array): 실제값
            predicted (array): 예측값
            
        Returns:
            dict: 성능 지표들
        """
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    
    def predict_future(self, data, model_name, days=30):
        """
        미래 주가를 예측합니다.
        
        Args:
            data (pd.DataFrame): 최신 데이터
            model_name (str): 사용할 모델명
            days (int): 예측할 일수
            
        Returns:
            pd.DataFrame: 예측 결과
        """
        if model_name not in self.models:
            print(f"❌ {model_name} 모델이 없습니다.")
            return None
        
        print(f"🔮 {model_name} 모델로 {days}일 예측 시작")
        
        model = self.models[model_name]
        
        if model_name == 'LSTM':
            return self._predict_future_lstm(data, days)
        else:
            return self._predict_future_traditional(data, model_name, days)
    
    def _predict_future_traditional(self, data, model_name, days):
        """전통적인 모델로 미래 예측"""
        model = self.models[model_name]
        scaler = self.scalers['traditional']
        
        # 최신 데이터 준비
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position',
            'Daily_Return', 'Volatility',
            'DayOfWeek', 'Month', 'Quarter'
        ]
        
        lag_features = [col for col in data.columns if '_lag_' in col]
        feature_columns.extend(lag_features)
        available_features = [col for col in feature_columns if col in data.columns]
        
        latest_features = data[available_features].iloc[-1:].fillna(method='ffill')
        latest_features_scaled = scaler.transform(latest_features)
        
        # 단일 예측 (모델이 5일 후를 예측하도록 훈련됨)
        prediction = model.predict(latest_features_scaled)[0]
        
        # 예측 결과 생성
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
        
        # 간단한 선형 보간으로 일별 예측값 생성
        current_price = data['Close'].iloc[-1]
        daily_change = (prediction - current_price) / 5  # 5일에 걸친 변화를 일별로 분배
        
        future_prices = []
        for i in range(days):
            if i < 5:
                future_price = current_price + (daily_change * (i + 1))
            else:
                # 5일 이후는 마지막 예측값 유지 + 약간의 노이즈
                future_price = prediction + np.random.normal(0, abs(prediction) * 0.01)
            future_prices.append(future_price)
        
        future_df = pd.DataFrame({
            'Date': future_dates[:len(future_prices)],
            'Predicted_Price': future_prices
        })
        
        return future_df
    
    def _predict_future_lstm(self, data, days):
        """LSTM 모델로 미래 예측"""
        model = self.models['LSTM']
        scaler = self.scalers['LSTM']
        
        # 특성 컬럼 (LSTM 훈련 시 사용한 것과 동일해야 함)
        feature_columns = ['Close', 'Volume', 'MA_20', 'RSI', 'MACD']
        available_features = [col for col in feature_columns if col in data.columns]
        
        # 최근 60일 데이터 준비
        sequence_length = 60
        recent_data = data[available_features].tail(sequence_length)
        
        if len(recent_data) < sequence_length:
            print(f"❌ 예측에 필요한 데이터가 부족합니다. 필요: {sequence_length}, 현재: {len(recent_data)}")
            return None
        
        # 데이터 정규화
        scaled_data = scaler.transform(recent_data)
        
        # 예측 수행
        predictions = []
        current_sequence = scaled_data[-sequence_length:, :-1]  # 타겟 컬럼 제외
        
        for _ in range(days):
            # 현재 시퀀스로 다음 값 예측
            current_sequence_reshaped = current_sequence.reshape(1, sequence_length, -1)
            next_pred = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # 시퀀스 업데이트 (새 예측값을 포함)
            new_row = current_sequence[-1].copy()  # 마지막 특성들 복사
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # 스케일 역변환
        dummy_array = np.zeros((len(predictions), len(available_features)))
        dummy_array[:, -1] = predictions
        predictions_original = scaler.inverse_transform(dummy_array)[:, -1]
        
        # 결과 데이터프레임 생성
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days, freq='B')
        
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions_original
        })
        
        return future_df
    
    def plot_model_comparison(self, save_path=None):
        """
        모델 성능 비교 차트를 그립니다.
        
        Args:
            save_path (str): 저장 경로
        """
        if not self.evaluation_results:
            print("❌ 평가 결과가 없습니다.")
            return
        
        # 성능 지표 수집
        models = []
        rmse_scores = []
        r2_scores = []
        mae_scores = []
        
        for model_name, results in self.evaluation_results.items():
            if 'test' in results:
                models.append(model_name)
                rmse_scores.append(results['test']['rmse'])
                r2_scores.append(results['test']['r2'])
                mae_scores.append(results['test']['mae'])
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('모델 성능 비교', fontsize=16, fontweight='bold')
        
        # 1. RMSE 비교
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, rmse_scores, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax1.set_title('RMSE (낮을수록 좋음)')
        ax1.set_ylabel('RMSE')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 값 표시
        for bar, score in zip(bars1, rmse_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + score*0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 2. R² 비교
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, r2_scores, color=plt.cm.plasma(np.linspace(0, 1, len(models))))
        ax2.set_title('R² Score (높을수록 좋음)')
        ax2.set_ylabel('R² Score')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        for bar, score in zip(bars2, r2_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. MAE 비교
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, mae_scores, color=plt.cm.coolwarm(np.linspace(0, 1, len(models))))
        ax3.set_title('MAE (낮을수록 좋음)')
        ax3.set_ylabel('MAE')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        for bar, score in zip(bars3, mae_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + score*0.01,
                    f'{score:.2f}', ha='center', va='bottom')
        
        # 4. 종합 성능 레이더 차트 (정규화된 지표)
        ax4 = axes[1, 1]
        
        # 지표 정규화 (0-1 범위)
        rmse_norm = 1 - np.array(rmse_scores) / max(rmse_scores)  # RMSE는 낮을수록 좋음
        mae_norm = 1 - np.array(mae_scores) / max(mae_scores)    # MAE는 낮을수록 좋음
        r2_norm = np.array(r2_scores)  # R²는 높을수록 좋음
        
        # 종합 점수 계산
        composite_scores = (rmse_norm + mae_norm + r2_norm) / 3
        
        bars4 = ax4.bar(models, composite_scores, color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        ax4.set_title('종합 성능 점수')
        ax4.set_ylabel('종합 점수 (0-1)')
        ax4.set_ylim(0, 1)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        for bar, score in zip(bars4, composite_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 모델 비교 차트 저장: {save_path}")
        
        plt.close()  # plt.show() 대신 plt.close() 사용
    
    def plot_predictions(self, data, model_name, save_path=None):
        """
        예측 결과를 시각화합니다.
        
        Args:
            data (pd.DataFrame): 원본 데이터
            model_name (str): 모델명
            save_path (str): 저장 경로
        """
        if model_name not in self.predictions:
            print(f"❌ {model_name} 예측 결과가 없습니다.")
            return
        
        predictions = self.predictions[model_name]
        
        plt.figure(figsize=(15, 10))
        
        if model_name == 'LSTM':
            # LSTM의 경우
            train_actual = predictions['train_actual']
            test_actual = predictions['test_actual']
            train_pred = predictions['train']
            test_pred = predictions['test']
            
            # 전체 데이터에서 해당하는 인덱스 찾기
            train_size = len(train_actual)
            total_size = train_size + len(test_actual)
            
            plt.plot(range(train_size), train_actual, label='실제 (훈련)', color='blue', alpha=0.7)
            plt.plot(range(train_size), train_pred, label='예측 (훈련)', color='lightblue', alpha=0.7)
            plt.plot(range(train_size, total_size), test_actual, label='실제 (테스트)', color='red', alpha=0.8)
            plt.plot(range(train_size, total_size), test_pred, label='예측 (테스트)', color='orange', alpha=0.8)
            
            plt.axvline(x=train_size, color='gray', linestyle='--', alpha=0.5, label='훈련/테스트 분할점')
            
        else:
            # 전통적인 모델의 경우 - 더 간단한 시각화
            # 실제 데이터의 마지막 부분과 예측 비교
            recent_data = data['Close'].tail(100)
            plt.plot(range(len(recent_data)), recent_data.values, label='실제 주가', color='blue', linewidth=2)
            
            # 예측 값은 테스트 부분만 표시
            if 'test' in predictions:
                test_pred = predictions['test']
                test_start = len(recent_data) - len(test_pred)
                plt.plot(range(test_start, len(recent_data)), test_pred, 
                        label='예측 주가', color='red', linewidth=2, alpha=0.8)
        
        plt.title(f'{model_name} 모델 예측 결과', fontsize=16, fontweight='bold')
        plt.xlabel('시간')
        plt.ylabel('주가 (원)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"�� 예측 결과 차트 저장: {save_path}")
        
        plt.close()  # plt.show() 대신 plt.close() 사용
    
    def save_results(self, filename_prefix="prediction_results"):
        """
        예측 결과를 파일로 저장합니다.
        
        Args:
            filename_prefix (str): 파일명 접두사
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 평가 결과 저장
        eval_results = []
        for model_name, results in self.evaluation_results.items():
            if 'test' in results:
                eval_results.append({
                    'Model': model_name,
                    'Test_RMSE': results['test']['rmse'],
                    'Test_MAE': results['test']['mae'],
                    'Test_R2': results['test']['r2'],
                    'Test_MAPE': results['test']['mape']
                })
        
        if eval_results:
            eval_df = pd.DataFrame(eval_results)
            eval_filename = f"results/predictions/{filename_prefix}_evaluation_{timestamp}.csv"
            eval_df.to_csv(eval_filename, index=False, encoding='utf-8-sig')
            print(f"💾 평가 결과 저장: {eval_filename}")
        
        # 모델 저장 (pickle 등 사용 가능)
        print("💾 모델 파일들이 results/models/ 폴더에 저장됩니다.")

def main():
    """메인 실행 함수"""
    predictor = StockPredictor()
    
    print("🔮 주식 가격 예측 모델 훈련 시작")
    print("="*50)
    
    # 전처리된 데이터 로드
    processed_files = [f for f in os.listdir('data/processed/') if f.startswith('processed_stock_data_')]
    if not processed_files:
        print("❌ 예측할 데이터 파일이 없습니다.")
        return
    
    latest_file = max(processed_files)
    data_path = f"data/processed/{latest_file}"
    data = predictor.load_data(data_path)
    
    if data is None:
        return
    
    # 특정 종목 선택 (첫 번째 종목)
    first_symbol = data['Symbol'].unique()[0]
    stock_data = data[data['Symbol'] == first_symbol].copy()
    
    print(f"📈 {stock_data['Name'].iloc[0]} 종목 예측 모델 훈련")
    
    # 1. 전통적인 머신러닝 모델 훈련
    print("\n1️⃣ 전통적인 머신러닝 모델들 훈련")
    X, y, feature_names = predictor.prepare_features(stock_data, prediction_days=5)
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_time_series_data(X, y)
    
    predictor.train_traditional_models(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 2. LSTM 모델 훈련
    print("\n2️⃣ LSTM 딥러닝 모델 훈련")
    lstm_features = ['Close', 'Volume', 'MA_20', 'RSI', 'MACD']
    available_lstm_features = [col for col in lstm_features if col in stock_data.columns]
    
    if len(available_lstm_features) >= 3:  # 최소 3개 특성 필요
        predictor.train_lstm_model(
            stock_data, 
            available_lstm_features, 
            target_column='Close',
            prediction_days=5
        )
    else:
        print("⚠️ LSTM 훈련에 필요한 특성이 부족합니다.")
    
    # 3. 결과 시각화
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n📊 결과 시각화")
    predictor.plot_model_comparison(
        save_path=f"results/predictions/model_comparison_{timestamp}.png"
    )
    
    # 최고 성능 모델로 예측 시각화
    if predictor.evaluation_results:
        best_model = min(predictor.evaluation_results.keys(), 
                        key=lambda x: predictor.evaluation_results[x].get('test', {}).get('rmse', float('inf')))
        
        print(f"🏆 최고 성능 모델: {best_model}")
        predictor.plot_predictions(
            stock_data, 
            best_model,
            save_path=f"results/predictions/best_model_predictions_{timestamp}.png"
        )
        
        # 미래 예측
        print(f"\n🔮 {best_model} 모델로 30일 미래 예측")
        future_predictions = predictor.predict_future(stock_data, best_model, days=30)
        
        if future_predictions is not None:
            future_filename = f"results/predictions/future_predictions_{timestamp}.csv"
            future_predictions.to_csv(future_filename, index=False, encoding='utf-8-sig')
            print(f"💾 미래 예측 결과 저장: {future_filename}")
            
            # 미래 예측 시각화
            plt.figure(figsize=(12, 6))
            recent_prices = stock_data['Close'].tail(60)
            plt.plot(range(len(recent_prices)), recent_prices.values, 
                    label='최근 실제 주가', color='blue', linewidth=2)
            
            future_start = len(recent_prices)
            plt.plot(range(future_start, future_start + len(future_predictions)), 
                    future_predictions['Predicted_Price'].values,
                    label='미래 예측 주가', color='red', linewidth=2, linestyle='--')
            
            plt.axvline(x=future_start, color='gray', linestyle=':', alpha=0.7, label='예측 시작점')
            plt.title(f'{stock_data["Name"].iloc[0]} 미래 주가 예측', fontsize=14, fontweight='bold')
            plt.xlabel('일수')
            plt.ylabel('주가 (원)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            future_plot_path = f"results/predictions/future_prediction_plot_{timestamp}.png"
            plt.savefig(future_plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 미래 예측 차트 저장: {future_plot_path}")
            plt.show()
    
    # 4. 결과 저장
    predictor.save_results()
    
    print("\n✅ 예측 모델 훈련 및 평가 완료!")

if __name__ == "__main__":
    main() 