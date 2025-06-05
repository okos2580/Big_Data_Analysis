"""
주식 데이터 분석 모듈
전처리된 데이터를 사용하여 다양한 분석을 수행합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class StockDataAnalyzer:
    def __init__(self):
        """데이터 분석기 초기화"""
        os.makedirs('results/analysis', exist_ok=True)
        
    def load_processed_data(self, filepath):
        """
        전처리된 데이터를 로드합니다.
        
        Args:
            filepath (str): 전처리된 데이터 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터
        """
        try:
            data = pd.read_csv(filepath, encoding='utf-8-sig', index_col=0, parse_dates=True)
            print(f"✓ 전처리된 데이터 로드 완료: {filepath}")
            return data
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            return None
    
    def basic_statistics(self, data):
        """
        기본 통계 분석을 수행합니다.
        
        Args:
            data (pd.DataFrame): 분석할 데이터
            
        Returns:
            dict: 분석 결과
        """
        results = {}
        
        print("📊 기본 통계 분석 수행 중...")
        
        # 1. 전체 데이터 요약
        results['data_summary'] = {
            '총 거래일수': len(data),
            '분석 기간': f"{data.index.min().strftime('%Y-%m-%d')} ~ {data.index.max().strftime('%Y-%m-%d')}",
            '종목 수': data['Symbol'].nunique(),
            '특성 수': len(data.columns)
        }
        
        # 2. 주요 지표별 기술통계
        key_metrics = ['Close', 'Volume', 'Daily_Return', 'RSI', 'Volatility']
        results['descriptive_stats'] = {}
        
        for metric in key_metrics:
            if metric in data.columns:
                results['descriptive_stats'][metric] = {
                    '평균': data[metric].mean(),
                    '중앙값': data[metric].median(),
                    '표준편차': data[metric].std(),
                    '최솟값': data[metric].min(),
                    '최댓값': data[metric].max(),
                    '1사분위수': data[metric].quantile(0.25),
                    '3사분위수': data[metric].quantile(0.75),
                    '왜도': stats.skew(data[metric].dropna()),
                    '첨도': stats.kurtosis(data[metric].dropna())
                }
        
        # 3. 종목별 기본 통계
        results['stock_summary'] = []
        for symbol in data['Symbol'].unique():
            stock_data = data[data['Symbol'] == symbol]
            
            # 수익률 계산
            start_price = stock_data['Close'].iloc[0]
            end_price = stock_data['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price * 100
            
            summary = {
                'Symbol': symbol,
                'Name': stock_data['Name'].iloc[0],
                '시작가': start_price,
                '종료가': end_price,
                '총 수익률(%)': total_return,
                '평균 일일수익률(%)': stock_data['Daily_Return'].mean() * 100,
                '변동성': stock_data['Daily_Return'].std() * np.sqrt(252),
                '최대값': stock_data['Close'].max(),
                '최솟값': stock_data['Close'].min(),
                '평균 거래량': stock_data['Volume'].mean(),
                '평균 RSI': stock_data['RSI'].mean()
            }
            results['stock_summary'].append(summary)
        
        results['stock_summary'] = pd.DataFrame(results['stock_summary'])
        
        print("✓ 기본 통계 분석 완료")
        return results
    
    def correlation_analysis(self, data):
        """
        상관관계 분석을 수행합니다.
        
        Args:
            data (pd.DataFrame): 분석할 데이터
            
        Returns:
            dict: 상관관계 분석 결과
        """
        print("🔗 상관관계 분석 수행 중...")
        
        results = {}
        
        # 수치형 데이터만 선택
        numeric_data = data.select_dtypes(include=[np.number])
        
        # 1. 전체 상관관계 매트릭스
        correlation_matrix = numeric_data.corr()
        results['correlation_matrix'] = correlation_matrix
        
        # 2. 주요 지표들 간의 상관관계
        key_indicators = ['Close', 'Volume', 'Daily_Return', 'RSI', 'MACD', 'BB_Position', 'Volatility']
        available_indicators = [col for col in key_indicators if col in numeric_data.columns]
        
        if available_indicators:
            key_correlations = numeric_data[available_indicators].corr()
            results['key_correlations'] = key_correlations
        
        # 3. 강한 상관관계 (|correlation| > 0.7) 찾기
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'Variable1': correlation_matrix.columns[i],
                        'Variable2': correlation_matrix.columns[j],
                        'Correlation': corr_value,
                        'Strength': 'Very Strong' if abs(corr_value) > 0.9 else 'Strong'
                    })
        
        results['strong_correlations'] = pd.DataFrame(strong_correlations)
        
        print("✓ 상관관계 분석 완료")
        return results
    
    def time_series_analysis(self, data):
        """
        시계열 분석을 수행합니다.
        
        Args:
            data (pd.DataFrame): 분석할 데이터
            
        Returns:
            dict: 시계열 분석 결과
        """
        print("📈 시계열 분석 수행 중...")
        
        results = {}
        
        # 1. 월별 분석
        monthly_stats = []
        for symbol in data['Symbol'].unique():
            stock_data = data[data['Symbol'] == symbol].copy()
            stock_data['YearMonth'] = stock_data.index.to_period('M')
            
            monthly_returns = stock_data.groupby('YearMonth')['Daily_Return'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            monthly_returns['Symbol'] = symbol
            monthly_returns['Name'] = stock_data['Name'].iloc[0]
            monthly_stats.append(monthly_returns)
        
        if monthly_stats:
            results['monthly_analysis'] = pd.concat(monthly_stats, ignore_index=True)
        
        # 2. 요일별 분석
        weekday_stats = []
        weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        
        for symbol in data['Symbol'].unique():
            stock_data = data[data['Symbol'] == symbol]
            
            for day in range(7):
                day_data = stock_data[stock_data['DayOfWeek'] == day]
                if len(day_data) > 0:
                    weekday_stats.append({
                        'Symbol': symbol,
                        'Name': stock_data['Name'].iloc[0],
                        'DayOfWeek': day,
                        'DayName': weekday_names[day],
                        '평균수익률': day_data['Daily_Return'].mean() * 100,
                        '수익률표준편차': day_data['Daily_Return'].std() * 100,
                        '거래일수': len(day_data),
                        '양의수익률비율': (day_data['Daily_Return'] > 0).sum() / len(day_data) * 100
                    })
        
        results['weekday_analysis'] = pd.DataFrame(weekday_stats)
        
        # 3. 변동성 분석
        volatility_stats = []
        for symbol in data['Symbol'].unique():
            stock_data = data[data['Symbol'] == symbol]
            
            # 월별 변동성
            monthly_vol = stock_data.groupby(stock_data.index.to_period('M'))['Daily_Return'].std() * np.sqrt(21)
            
            volatility_stats.append({
                'Symbol': symbol,
                'Name': stock_data['Name'].iloc[0],
                '연평균변동성': stock_data['Daily_Return'].std() * np.sqrt(252),
                '최대월변동성': monthly_vol.max(),
                '최소월변동성': monthly_vol.min(),
                '변동성평균': monthly_vol.mean(),
                '변동성표준편차': monthly_vol.std()
            })
        
        results['volatility_analysis'] = pd.DataFrame(volatility_stats)
        
        print("✓ 시계열 분석 완료")
        return results
    
    def performance_analysis(self, data):
        """
        성과 분석을 수행합니다.
        
        Args:
            data (pd.DataFrame): 분석할 데이터
            
        Returns:
            dict: 성과 분석 결과
        """
        print("🎯 성과 분석 수행 중...")
        
        results = {}
        performance_metrics = []
        
        for symbol in data['Symbol'].unique():
            stock_data = data[data['Symbol'] == symbol].copy()
            returns = stock_data['Daily_Return'].dropna()
            
            if len(returns) == 0:
                continue
            
            # 기본 성과 지표 계산
            total_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
            annualized_return = returns.mean() * 252 * 100
            annualized_volatility = returns.std() * np.sqrt(252) * 100
            
            # 샤프 비율 (무위험 수익률을 0으로 가정)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
            
            # 최대 낙폭 (Maximum Drawdown)
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # VaR (Value at Risk) 95% 신뢰구간
            var_95 = np.percentile(returns, 5) * 100
            
            # 승률 (양의 수익률 비율)
            win_rate = (returns > 0).sum() / len(returns) * 100
            
            # 수익/손실 비율
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            avg_gain = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
            avg_loss = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
            profit_loss_ratio = abs(avg_gain / avg_loss) if avg_loss != 0 else 0
            
            performance_metrics.append({
                'Symbol': symbol,
                'Name': stock_data['Name'].iloc[0],
                '총수익률(%)': total_return,
                '연환산수익률(%)': annualized_return,
                '연환산변동성(%)': annualized_volatility,
                '샤프비율': sharpe_ratio,
                '최대낙폭(%)': max_drawdown,
                'VaR_95(%)': var_95,
                '승률(%)': win_rate,
                '평균수익(%)': avg_gain,
                '평균손실(%)': avg_loss,
                '수익손실비율': profit_loss_ratio
            })
        
        results['performance_metrics'] = pd.DataFrame(performance_metrics)
        
        # 성과 순위
        if not results['performance_metrics'].empty:
            # 샤프 비율 기준 순위
            results['sharpe_ranking'] = results['performance_metrics'].sort_values('샤프비율', ascending=False)
            
            # 총 수익률 기준 순위
            results['return_ranking'] = results['performance_metrics'].sort_values('총수익률(%)', ascending=False)
            
            # 위험 조정 수익률 기준 (수익률/변동성)
            results['performance_metrics']['위험조정수익률'] = (
                results['performance_metrics']['연환산수익률(%)'] / 
                results['performance_metrics']['연환산변동성(%)']
            )
            results['risk_adjusted_ranking'] = results['performance_metrics'].sort_values('위험조정수익률', ascending=False)
        
        print("✓ 성과 분석 완료")
        return results
    
    def technical_analysis(self, data):
        """
        기술적 분석을 수행합니다.
        
        Args:
            data (pd.DataFrame): 분석할 데이터
            
        Returns:
            dict: 기술적 분석 결과
        """
        print("📊 기술적 분석 수행 중...")
        
        results = {}
        technical_signals = []
        
        for symbol in data['Symbol'].unique():
            stock_data = data[data['Symbol'] == symbol].copy()
            latest_data = stock_data.iloc[-1]  # 최신 데이터
            
            # RSI 신호
            rsi_signal = '과매수' if latest_data['RSI'] > 70 else '과매도' if latest_data['RSI'] < 30 else '중립'
            
            # MACD 신호
            macd_signal = '매수' if latest_data['MACD'] > latest_data['MACD_Signal'] else '매도'
            
            # 볼린저 밴드 신호
            bb_position = latest_data['BB_Position']
            bb_signal = '상단돌파' if bb_position > 1 else '하단돌파' if bb_position < 0 else '중간권'
            
            # 이동평균선 신호
            ma_signal = []
            if latest_data['Close'] > latest_data['MA_20']:
                ma_signal.append('20일선 상회')
            if latest_data['Close'] > latest_data['MA_50']:
                ma_signal.append('50일선 상회')
            
            ma_signal_text = ', '.join(ma_signal) if ma_signal else '이동평균선 하회'
            
            # 거래량 분석
            volume_ratio = latest_data['Volume'] / latest_data['Volume_MA']
            volume_signal = '급증' if volume_ratio > 2 else '증가' if volume_ratio > 1.5 else '보통'
            
            technical_signals.append({
                'Symbol': symbol,
                'Name': stock_data['Name'].iloc[0],
                '현재가': latest_data['Close'],
                'RSI': latest_data['RSI'],
                'RSI신호': rsi_signal,
                'MACD신호': macd_signal,
                '볼린저밴드신호': bb_signal,
                '이동평균선신호': ma_signal_text,
                '거래량신호': volume_signal,
                '거래량비율': volume_ratio
            })
        
        results['technical_signals'] = pd.DataFrame(technical_signals)
        
        # 전체 시장 기술적 지표 요약
        market_summary = {
            '평균_RSI': data['RSI'].mean(),
            'RSI_과매수_종목수': len(data[data['RSI'] > 70]['Symbol'].unique()),
            'RSI_과매도_종목수': len(data[data['RSI'] < 30]['Symbol'].unique()),
            '평균_변동성': data['Volatility'].mean(),
            '고변동성_종목수': len(data[data['Volatility'] > data['Volatility'].quantile(0.8)]['Symbol'].unique())
        }
        
        results['market_summary'] = market_summary
        
        print("✓ 기술적 분석 완료")
        return results
    
    def save_analysis_results(self, results, filename_prefix="analysis_results"):
        """
        분석 결과를 파일로 저장합니다.
        
        Args:
            results (dict): 분석 결과
            filename_prefix (str): 파일명 접두사
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for analysis_type, result in results.items():
            if isinstance(result, pd.DataFrame):
                filename = f"results/analysis/{filename_prefix}_{analysis_type}_{timestamp}.csv"
                result.to_csv(filename, encoding='utf-8-sig', index=False)
                print(f"💾 {analysis_type} 결과 저장: {filename}")
        
        # 전체 결과 요약을 텍스트 파일로 저장
        summary_filename = f"results/analysis/{filename_prefix}_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("=== 주식 데이터 분석 결과 요약 ===\n\n")
            f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for analysis_type, result in results.items():
                f.write(f"## {analysis_type}\n")
                if isinstance(result, dict):
                    for key, value in result.items():
                        f.write(f"{key}: {value}\n")
                elif isinstance(result, pd.DataFrame):
                    f.write(f"데이터 형태: DataFrame ({len(result)} 행, {len(result.columns)} 열)\n")
                    f.write(f"컬럼: {', '.join(result.columns)}\n")
                f.write("\n")
        
        print(f"📄 분석 요약 저장: {summary_filename}")

def main():
    """메인 실행 함수"""
    analyzer = StockDataAnalyzer()
    
    print("📊 주식 데이터 분석 시작")
    print("="*50)
    
    # 최근 전처리된 데이터 파일 찾기
    processed_files = [f for f in os.listdir('data/processed/') if f.startswith('processed_stock_data_')]
    if not processed_files:
        print("❌ 분석할 전처리된 데이터 파일이 없습니다. 먼저 data_preprocessing.py를 실행하세요.")
        return
    
    latest_file = max(processed_files)
    filepath = f"data/processed/{latest_file}"
    
    # 데이터 로드
    data = analyzer.load_processed_data(filepath)
    if data is None:
        return
    
    # 전체 분석 실행
    all_results = {}
    
    # 1. 기본 통계 분석
    all_results.update(analyzer.basic_statistics(data))
    
    # 2. 상관관계 분석
    correlation_results = analyzer.correlation_analysis(data)
    all_results['correlation_analysis'] = correlation_results
    
    # 3. 시계열 분석
    timeseries_results = analyzer.time_series_analysis(data)
    all_results.update(timeseries_results)
    
    # 4. 성과 분석
    performance_results = analyzer.performance_analysis(data)
    all_results.update(performance_results)
    
    # 5. 기술적 분석
    technical_results = analyzer.technical_analysis(data)
    all_results.update(technical_results)
    
    # 결과 저장
    analyzer.save_analysis_results(all_results)
    
    print("\n📈 주요 분석 결과:")
    print(f"  - 분석 종목 수: {data['Symbol'].nunique()}")
    print(f"  - 분석 기간: {data.index.min().strftime('%Y-%m-%d')} ~ {data.index.max().strftime('%Y-%m-%d')}")
    
    if 'performance_metrics' in all_results:
        best_performer = all_results['performance_metrics'].loc[all_results['performance_metrics']['총수익률(%)'].idxmax()]
        worst_performer = all_results['performance_metrics'].loc[all_results['performance_metrics']['총수익률(%)'].idxmin()]
        
        print(f"  - 최고 수익률: {best_performer['Name']} ({best_performer['총수익률(%)']:.2f}%)")
        print(f"  - 최저 수익률: {worst_performer['Name']} ({worst_performer['총수익률(%)']:.2f}%)")
    
    print("\n✅ 데이터 분석 완료!")

if __name__ == "__main__":
    main() 