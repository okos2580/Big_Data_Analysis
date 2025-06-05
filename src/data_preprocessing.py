"""
주식 데이터 전처리 모듈
수집된 원시 데이터를 분석에 적합한 형태로 정제하고 변환합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

class StockDataPreprocessor:
    def __init__(self):
        """데이터 전처리기 초기화"""
        os.makedirs('data/processed', exist_ok=True)
        
    def load_data(self, filepath):
        """
        CSV 파일에서 데이터를 로드합니다.
        
        Args:
            filepath (str): CSV 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        try:
            data = pd.read_csv(filepath, encoding='utf-8-sig', index_col=0)
            # 인덱스를 datetime으로 변환
            data.index = pd.to_datetime(data.index)
            print(f"✓ 데이터 로드 완료: {filepath}")
            print(f"  - 행 수: {len(data)}")
            print(f"  - 열 수: {len(data.columns)}")
            return data
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            return None
    
    def clean_data(self, data):
        """
        기본적인 데이터 정제를 수행합니다.
        
        Args:
            data (pd.DataFrame): 원시 데이터
            
        Returns:
            pd.DataFrame: 정제된 데이터
        """
        cleaned_data = data.copy()
        
        # 결측치 확인
        missing_values = cleaned_data.isnull().sum()
        if missing_values.sum() > 0:
            print("결측치 발견:")
            print(missing_values[missing_values > 0])
            
            # 수치형 컬럼의 결측치를 이전 값으로 채우기
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(method='ffill')
            
            # 여전히 결측치가 있다면 평균값으로 채우기
            cleaned_data[numeric_columns] = cleaned_data[numeric_columns].fillna(cleaned_data[numeric_columns].mean())
            
        # 중복 행 제거
        duplicate_count = cleaned_data.duplicated().sum()
        if duplicate_count > 0:
            print(f"중복 행 {duplicate_count}개 제거")
            cleaned_data = cleaned_data.drop_duplicates()
            
        # 음수 거래량 제거 (있다면)
        if 'Volume' in cleaned_data.columns:
            negative_volume = (cleaned_data['Volume'] < 0).sum()
            if negative_volume > 0:
                print(f"음수 거래량 {negative_volume}개 제거")
                cleaned_data = cleaned_data[cleaned_data['Volume'] >= 0]
        
        print("✓ 기본 데이터 정제 완료")
        return cleaned_data
    
    def detect_outliers(self, data, column, method='iqr'):
        """
        이상치를 탐지합니다.
        
        Args:
            data (pd.DataFrame): 데이터
            column (str): 이상치를 찾을 컬럼
            method (str): 탐지 방법 ('iqr' 또는 'zscore')
            
        Returns:
            pd.Series: 이상치 여부 (True/False)
        """
        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            outliers = z_scores > 3
            
        return outliers
    
    def calculate_technical_indicators(self, data):
        """
        기술적 지표를 계산합니다.
        
        Args:
            data (pd.DataFrame): 주식 데이터
            
        Returns:
            pd.DataFrame: 기술적 지표가 추가된 데이터
        """
        df = data.copy()
        
        # 1. 이동평균선 (Moving Averages)
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # 2. 지수 이동평균선 (EMA)
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # 3. MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 4. RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 5. 볼린저 밴드 (Bollinger Bands)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # 6. 일일 수익률
        df['Daily_Return'] = df['Close'].pct_change()
        
        # 7. 변동성 (20일 이동 표준편차)
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)
        
        # 8. 거래량 이동평균
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # 9. 가격 변화량
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
        
        # 10. 고가-저가 범위
        df['High_Low_Range'] = df['High'] - df['Low']
        df['High_Low_Range_Pct'] = (df['High'] - df['Low']) / df['Low'] * 100
        
        print("✓ 기술적 지표 계산 완료")
        return df
    
    def add_time_features(self, data):
        """
        시간 관련 특성을 추가합니다.
        
        Args:
            data (pd.DataFrame): 데이터
            
        Returns:
            pd.DataFrame: 시간 특성이 추가된 데이터
        """
        df = data.copy()
        
        # 날짜 인덱스에서 시간 특성 추출
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek  # 0: 월요일, 6: 일요일
        df['Quarter'] = df.index.quarter
        
        # 요일명 추가
        df['DayName'] = df.index.day_name()
        
        # 월 초/중/말 구분
        df['MonthPeriod'] = df['Day'].apply(lambda x: 'Early' if x <= 10 else 'Mid' if x <= 20 else 'Late')
        
        print("✓ 시간 특성 추가 완료")
        return df
    
    def create_lag_features(self, data, columns, lags=[1, 2, 3, 5]):
        """
        지연(lag) 특성을 생성합니다.
        
        Args:
            data (pd.DataFrame): 데이터
            columns (list): 지연 특성을 만들 컬럼들
            lags (list): 지연 기간들
            
        Returns:
            pd.DataFrame: 지연 특성이 추가된 데이터
        """
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        print(f"✓ 지연 특성 생성 완료 ({len(columns)}개 컬럼 × {len(lags)}개 지연)")
        return df
    
    def normalize_data(self, data, method='minmax'):
        """
        데이터를 정규화합니다.
        
        Args:
            data (pd.DataFrame): 데이터
            method (str): 정규화 방법 ('minmax' 또는 'zscore')
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        df = data.copy()
        
        # 수치형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            # Min-Max 정규화 (0-1 범위)
            df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].min()) / (df[numeric_columns].max() - df[numeric_columns].min())
        
        elif method == 'zscore':
            # Z-score 정규화 (평균 0, 표준편차 1)
            df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        
        print(f"✓ 데이터 정규화 완료 ({method} 방법)")
        return df
    
    def preprocess_full_pipeline(self, data, include_lags=True, normalize=False):
        """
        전체 전처리 파이프라인을 실행합니다.
        
        Args:
            data (pd.DataFrame): 원시 데이터
            include_lags (bool): 지연 특성 포함 여부
            normalize (bool): 정규화 수행 여부
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        print("🔧 전체 데이터 전처리 시작")
        print("="*50)
        
        # 1. 기본 정제
        processed_data = self.clean_data(data)
        
        # 2. 기술적 지표 계산
        processed_data = self.calculate_technical_indicators(processed_data)
        
        # 3. 시간 특성 추가
        processed_data = self.add_time_features(processed_data)
        
        # 4. 지연 특성 생성 (선택적)
        if include_lags:
            lag_columns = ['Close', 'Volume', 'Daily_Return', 'RSI']
            processed_data = self.create_lag_features(processed_data, lag_columns)
        
        # 5. 정규화 (선택적)
        if normalize:
            processed_data = self.normalize_data(processed_data, method='minmax')
        
        # 결측치 제거 (기술적 지표 계산으로 인한 초기 NaN 값들)
        processed_data = processed_data.dropna()
        
        print(f"✅ 전처리 완료: {len(processed_data)}행, {len(processed_data.columns)}열")
        return processed_data
    
    def save_processed_data(self, data, filename):
        """
        전처리된 데이터를 저장합니다.
        
        Args:
            data (pd.DataFrame): 전처리된 데이터
            filename (str): 저장할 파일명
        """
        filepath = f"data/processed/{filename}"
        data.to_csv(filepath, encoding='utf-8-sig')
        print(f"💾 전처리된 데이터 저장: {filepath}")
        return filepath

def main():
    """메인 실행 함수"""
    preprocessor = StockDataPreprocessor()
    
    print("🔧 주식 데이터 전처리 시작")
    print("="*50)
    
    # 최근 생성된 데이터 파일 찾기
    raw_files = [f for f in os.listdir('data/raw/') if f.startswith('stock_data_combined_')]
    if not raw_files:
        print("❌ 처리할 원시 데이터 파일이 없습니다. 먼저 data_collection.py를 실행하세요.")
        return
    
    latest_file = max(raw_files)
    filepath = f"data/raw/{latest_file}"
    
    # 데이터 로드
    raw_data = preprocessor.load_data(filepath)
    if raw_data is None:
        return
    
    # 전체 전처리 실행
    processed_data = preprocessor.preprocess_full_pipeline(
        raw_data, 
        include_lags=True, 
        normalize=False
    )
    
    # 전처리된 데이터 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_filename = f"processed_stock_data_{timestamp}.csv"
    preprocessor.save_processed_data(processed_data, processed_filename)
    
    # 데이터 요약 정보 출력
    print("\n📊 전처리된 데이터 요약:")
    print(f"  - 기간: {processed_data.index.min()} ~ {processed_data.index.max()}")
    print(f"  - 종목 수: {processed_data['Symbol'].nunique()}")
    print(f"  - 총 데이터 포인트: {len(processed_data)}")
    print(f"  - 특성 수: {len(processed_data.columns)}")
    
    print("\n✅ 데이터 전처리 완료!")

if __name__ == "__main__":
    main() 