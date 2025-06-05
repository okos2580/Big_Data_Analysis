"""
주식 데이터 수집 모듈
yfinance를 사용하여 한국 주식 시장 데이터를 수집합니다.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class StockDataCollector:
    def __init__(self):
        """주식 데이터 수집기 초기화"""
        # 주요 한국 주식 종목들 (코스피 대형주)
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
        
        # 데이터 저장 폴더 생성
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
    def get_stock_data(self, symbol, period="1y"):
        """
        개별 주식 데이터를 가져옵니다.
        
        Args:
            symbol (str): 주식 심볼 (예: '005930.KS')
            period (str): 데이터 기간 ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        
        Returns:
            pd.DataFrame: 주식 데이터
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"경고: {symbol}에 대한 데이터가 없습니다.")
                return None
                
            # 데이터에 종목 정보 추가
            data['Symbol'] = symbol
            data['Name'] = self.korean_stocks.get(symbol, symbol)
            
            print(f"✓ {self.korean_stocks.get(symbol, symbol)} 데이터 수집 완료")
            return data
            
        except Exception as e:
            print(f"오류 발생 ({symbol}): {str(e)}")
            return None
    
    def get_multiple_stocks_data(self, symbols=None, period="1y"):
        """
        여러 주식의 데이터를 한번에 가져옵니다.
        
        Args:
            symbols (list): 주식 심볼 리스트. None이면 기본 종목들 사용
            period (str): 데이터 기간
            
        Returns:
            dict: 주식 심볼을 키로 하는 데이터프레임 딕셔너리
        """
        if symbols is None:
            symbols = list(self.korean_stocks.keys())
            
        stock_data = {}
        
        print(f"📊 {len(symbols)}개 종목 데이터 수집 시작...")
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if data is not None:
                stock_data[symbol] = data
                
        print(f"✅ 총 {len(stock_data)}개 종목 데이터 수집 완료")
        return stock_data
    
    def get_market_index_data(self, period="1y"):
        """
        시장 지수 데이터를 가져옵니다.
        
        Args:
            period (str): 데이터 기간
            
        Returns:
            dict: 지수 데이터
        """
        indices = {
            '^KS11': 'KOSPI',
            '^KQ11': 'KOSDAQ',
            '^IXIC': 'NASDAQ',
            '^GSPC': 'S&P 500'
        }
        
        index_data = {}
        
        print("📈 시장 지수 데이터 수집 중...")
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty:
                    data['Symbol'] = symbol
                    data['Name'] = name
                    index_data[symbol] = data
                    print(f"✓ {name} 데이터 수집 완료")
            except Exception as e:
                print(f"오류 발생 ({name}): {str(e)}")
                
        return index_data
    
    def save_data_to_csv(self, stock_data, filename_prefix="stock_data"):
        """
        수집한 데이터를 CSV 파일로 저장합니다.
        
        Args:
            stock_data (dict): 주식 데이터 딕셔너리
            filename_prefix (str): 파일명 접두사
        """
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
        
        print(f"💾 데이터 저장 완료: {len(stock_data)}개 파일")
        return combined_filename
    
    def get_company_info(self, symbol):
        """
        기업 정보를 가져옵니다.
        
        Args:
            symbol (str): 주식 심볼
            
        Returns:
            dict: 기업 정보
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_info = {
                'symbol': symbol,
                'name': info.get('longName', self.korean_stocks.get(symbol, symbol)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'website': info.get('website', 'N/A')
            }
            
            return company_info
            
        except Exception as e:
            print(f"기업 정보 수집 오류 ({symbol}): {str(e)}")
            return None

def main():
    """메인 실행 함수"""
    collector = StockDataCollector()
    
    print("🚀 한국 주식 시장 데이터 수집 시작")
    print("="*50)
    
    # 1. 주식 데이터 수집
    stock_data = collector.get_multiple_stocks_data(period="2y")
    
    # 2. 시장 지수 데이터 수집
    index_data = collector.get_market_index_data(period="2y")
    
    # 3. 데이터 저장
    if stock_data:
        filename = collector.save_data_to_csv(stock_data)
        print(f"📁 메인 데이터 파일: {filename}")
    
    if index_data:
        index_filename = collector.save_data_to_csv(index_data, "market_index")
        print(f"📁 지수 데이터 파일: {index_filename}")
    
    # 4. 기업 정보 수집
    print("\n🏢 기업 정보 수집 중...")
    company_infos = []
    for symbol in collector.korean_stocks.keys():
        info = collector.get_company_info(symbol)
        if info:
            company_infos.append(info)
    
    if company_infos:
        company_df = pd.DataFrame(company_infos)
        company_filename = f"data/raw/company_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        company_df.to_csv(company_filename, encoding='utf-8-sig', index=False)
        print(f"📁 기업 정보 파일: {company_filename}")
    
    print("\n✅ 데이터 수집 완료!")

if __name__ == "__main__":
    main() 