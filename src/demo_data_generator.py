"""
데모용 주식 데이터 생성기
실제 API 대신 가상의 주식 데이터를 생성하여 프로젝트 시연용으로 사용합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DemoStockDataGenerator:
    def __init__(self):
        """데모 데이터 생성기 초기화"""
        self.korean_stocks = {
            '005930.KS': '삼성전자',
            '000660.KS': 'SK하이닉스', 
            '035420.KS': 'NAVER',
            '207940.KS': '삼성바이오로직스',
            '006400.KS': '삼성SDI',
            '051910.KS': 'LG화학',
            '035720.KS': '카카오',
            '068270.KS': '셀트리온',
            '012330.KS': '현대모비스',
            '028260.KS': '삼성물산'
        }
        
        # 각 종목의 기준 가격
        self.base_prices = {
            '005930.KS': 70000,   # 삼성전자
            '000660.KS': 120000,  # SK하이닉스
            '035420.KS': 180000,  # NAVER
            '207940.KS': 850000,  # 삼성바이오로직스
            '006400.KS': 420000,  # 삼성SDI
            '051910.KS': 650000,  # LG화학
            '035720.KS': 80000,   # 카카오
            '068270.KS': 190000,  # 셀트리온
            '012330.KS': 260000,  # 현대모비스
            '028260.KS': 140000   # 삼성물산
        }
        
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
    def generate_stock_data(self, symbol, days=500):
        """
        가상의 주식 데이터를 생성합니다.
        
        Args:
            symbol (str): 주식 심볼
            days (int): 생성할 일수
            
        Returns:
            pd.DataFrame: 가상 주식 데이터
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 날짜 인덱스 생성 (주말 제외)
        dates = pd.bdate_range(start=start_date, end=end_date)
        
        # 기준 가격
        base_price = self.base_prices.get(symbol, 100000)
        
        # 주가 데이터 생성 (기하 브라운 운동 모델 사용)
        np.random.seed(42)  # 재현 가능한 결과를 위해
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 일일 수익률
        
        # 주가 계산
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # 첫 번째 가격 제거
        
        # OHLCV 데이터 생성
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # 일중 변동성 추가
            volatility = np.random.uniform(0.01, 0.03)
            high = close * (1 + volatility)
            low = close * (1 - volatility)
            
            # 시가는 전날 종가 기준으로 조정
            if i == 0:
                open_price = close
            else:
                open_change = np.random.normal(0, 0.01)
                open_price = prices[i-1] * (1 + open_change)
            
            # 거래량 생성 (가격 변동과 음의 상관관계)
            price_change = abs(close - open_price) / open_price
            base_volume = np.random.uniform(1000000, 5000000)
            volume = int(base_volume * (1 + price_change * 3))
            
            data.append({
                'Date': date,
                'Open': max(low, min(high, open_price)),
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # 종목 정보 추가
        df['Symbol'] = symbol
        df['Name'] = self.korean_stocks.get(symbol, symbol)
        
        print(f"✓ {self.korean_stocks.get(symbol, symbol)} 가상 데이터 생성 완료")
        
        return df
    
    def generate_all_stocks_data(self):
        """모든 종목의 가상 데이터를 생성합니다."""
        stock_data = {}
        
        print("📊 10개 종목 가상 데이터 생성 시작...")
        
        for symbol in self.korean_stocks.keys():
            data = self.generate_stock_data(symbol)
            stock_data[symbol] = data
        
        print(f"✅ 총 {len(stock_data)}개 종목 가상 데이터 생성 완료")
        return stock_data
    
    def generate_market_index_data(self):
        """시장 지수 가상 데이터를 생성합니다."""
        indices = {
            '^KS11': 'KOSPI',
            '^KQ11': 'KOSDAQ',
            '^IXIC': 'NASDAQ',
            '^GSPC': 'S&P 500'
        }
        
        base_prices = {
            '^KS11': 2500,
            '^KQ11': 900,
            '^IXIC': 15000,
            '^GSPC': 4500
        }
        
        index_data = {}
        
        print("📈 시장 지수 가상 데이터 생성 중...")
        
        for symbol, name in indices.items():
            # 지수는 주식보다 변동성이 낮음
            end_date = datetime.now()
            start_date = end_date - timedelta(days=500)
            dates = pd.bdate_range(start=start_date, end=end_date)
            
            base_price = base_prices.get(symbol, 2500)
            np.random.seed(hash(symbol) % 1000)  # 각 지수마다 다른 시드
            
            returns = np.random.normal(0.0003, 0.01, len(dates))
            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            prices = prices[1:]
            
            data = []
            for date, close in zip(dates, prices):
                volatility = np.random.uniform(0.005, 0.015)
                high = close * (1 + volatility)
                low = close * (1 - volatility)
                open_price = close * (1 + np.random.normal(0, 0.005))
                
                data.append({
                    'Date': date,
                    'Open': max(low, min(high, open_price)),
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': int(np.random.uniform(500000, 2000000))
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            df['Symbol'] = symbol
            df['Name'] = name
            
            index_data[symbol] = df
            print(f"✓ {name} 가상 데이터 생성 완료")
        
        return index_data
    
    def save_data_to_csv(self, stock_data, filename_prefix="stock_data"):
        """생성한 데이터를 CSV 파일로 저장합니다."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 개별 종목 데이터 저장
        for symbol, data in stock_data.items():
            company_name = self.korean_stocks.get(symbol, symbol.replace('.KS', ''))
            filename = f"data/raw/{filename_prefix}_{company_name}_{timestamp}.csv"
            data.to_csv(filename, encoding='utf-8-sig')
            
        # 통합 데이터 저장
        combined_data = pd.concat(stock_data.values(), ignore_index=True)
        combined_filename = f"data/raw/{filename_prefix}_combined_{timestamp}.csv"
        combined_data.to_csv(combined_filename, encoding='utf-8-sig')
        
        print(f"💾 가상 데이터 저장 완료: {len(stock_data)}개 파일")
        return combined_filename
    
    def generate_company_info(self):
        """기업 정보 가상 데이터를 생성합니다."""
        company_infos = []
        
        sectors = ['Technology', 'Chemical', 'Automobile', 'Bio', 'Internet']
        industries = ['Semiconductors', 'Chemical', 'Auto Parts', 'Biotechnology', 'Internet Services']
        
        for i, (symbol, name) in enumerate(self.korean_stocks.items()):
            info = {
                'symbol': symbol,
                'name': name,
                'sector': sectors[i % len(sectors)],
                'industry': industries[i % len(industries)],
                'market_cap': np.random.randint(10000000, 100000000) * 1000000,  # 10조-100조
                'employees': np.random.randint(10000, 100000),
                'website': f'https://www.{name.lower().replace(" ", "")}.com'
            }
            company_infos.append(info)
        
        company_df = pd.DataFrame(company_infos)
        company_filename = f"data/raw/company_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        company_df.to_csv(company_filename, encoding='utf-8-sig', index=False)
        
        print(f"📁 기업 정보 가상 데이터 파일: {company_filename}")
        return company_filename

def main():
    """메인 실행 함수"""
    generator = DemoStockDataGenerator()
    
    print("🚀 가상 주식 데이터 생성 시작")
    print("="*50)
    
    # 1. 주식 데이터 생성
    stock_data = generator.generate_all_stocks_data()
    
    # 2. 시장 지수 데이터 생성
    index_data = generator.generate_market_index_data()
    
    # 3. 데이터 저장
    if stock_data:
        filename = generator.save_data_to_csv(stock_data)
        print(f"📁 메인 데이터 파일: {filename}")
    
    if index_data:
        index_filename = generator.save_data_to_csv(index_data, "market_index")
        print(f"📁 지수 데이터 파일: {index_filename}")
    
    # 4. 기업 정보 생성
    print("\n🏢 기업 정보 가상 데이터 생성 중...")
    company_filename = generator.generate_company_info()
    
    print("\n✅ 가상 데이터 생성 완료!")
    print("💡 이제 다른 모듈들을 실행할 수 있습니다.")

if __name__ == "__main__":
    main() 