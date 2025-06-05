"""
주식 데이터 시각화 모듈
분석된 데이터를 다양한 차트와 그래프로 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 사용
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import koreanize_matplotlib
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class StockVisualization:
    def __init__(self):
        """시각화 객체 초기화"""
        # 결과 저장 폴더 생성
        os.makedirs('results/plots', exist_ok=True)
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 색상 팔레트 설정
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.style_settings = {
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11
        }
        
        # matplotlib 스타일 적용
        for key, value in self.style_settings.items():
            plt.rcParams[key] = value
    
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
            print(f"✓ 시각화용 데이터 로드 완료: {filepath}")
            return data
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            return None
    
    def plot_stock_prices(self, data, symbols=None, save_path=None):
        """
        주식 가격 추이를 시각화합니다.
        
        Args:
            data (pd.DataFrame): 주식 데이터
            symbols (list): 표시할 종목 심볼 (None이면 모든 종목)
            save_path (str): 저장 경로
        """
        if symbols is None:
            symbols = data['Symbol'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('주식 가격 추이 분석', fontsize=16, fontweight='bold')
        
        # 1. 종목별 가격 추이
        ax1 = axes[0, 0]
        for i, symbol in enumerate(symbols[:5]):  # 최대 5개 종목
            stock_data = data[data['Symbol'] == symbol]
            if not stock_data.empty:
                ax1.plot(stock_data.index, stock_data['Close'], 
                        label=stock_data['Name'].iloc[0], 
                        color=self.colors[i], linewidth=2)
        
        ax1.set_title('종목별 주가 추이')
        ax1.set_xlabel('날짜')
        ax1.set_ylabel('종가 (원)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 정규화된 가격 비교 (시작점을 100으로 설정)
        ax2 = axes[0, 1]
        for i, symbol in enumerate(symbols[:5]):
            stock_data = data[data['Symbol'] == symbol]
            if not stock_data.empty:
                normalized_price = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
                ax2.plot(stock_data.index, normalized_price, 
                        label=stock_data['Name'].iloc[0], 
                        color=self.colors[i], linewidth=2)
        
        ax2.set_title('정규화된 주가 비교 (시작점=100)')
        ax2.set_xlabel('날짜')
        ax2.set_ylabel('정규화된 가격')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 거래량 추이
        ax3 = axes[1, 0]
        for i, symbol in enumerate(symbols[:3]):  # 가독성을 위해 3개만
            stock_data = data[data['Symbol'] == symbol]
            if not stock_data.empty:
                ax3.plot(stock_data.index, stock_data['Volume'], 
                        label=stock_data['Name'].iloc[0], 
                        color=self.colors[i], linewidth=1.5)
        
        ax3.set_title('거래량 추이')
        ax3.set_xlabel('날짜')
        ax3.set_ylabel('거래량')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 일일 수익률 분포
        ax4 = axes[1, 1]
        returns_data = []
        labels = []
        for symbol in symbols[:5]:
            stock_data = data[data['Symbol'] == symbol]
            if not stock_data.empty:
                returns_data.append(stock_data['Daily_Return'].dropna() * 100)
                labels.append(stock_data['Name'].iloc[0])
        
        ax4.boxplot(returns_data, labels=labels)
        ax4.set_title('일일 수익률 분포')
        ax4.set_xlabel('종목')
        ax4.set_ylabel('일일 수익률 (%)')
        ax4.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 주가 추이 차트 저장: {save_path}")
        
        plt.close()
    
    def plot_technical_indicators(self, data, symbol, save_path=None):
        """
        특정 종목의 기술적 지표를 시각화합니다.
        
        Args:
            data (pd.DataFrame): 주식 데이터
            symbol (str): 종목 심볼
            save_path (str): 저장 경로
        """
        stock_data = data[data['Symbol'] == symbol].copy()
        if stock_data.empty:
            print(f"❌ {symbol} 데이터가 없습니다.")
            return
        
        company_name = stock_data['Name'].iloc[0]
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        fig.suptitle(f'{company_name} 기술적 지표 분석', fontsize=16, fontweight='bold')
        
        # 1. 주가와 이동평균선
        ax1 = axes[0]
        ax1.plot(stock_data.index, stock_data['Close'], label='종가', color='black', linewidth=2)
        ax1.plot(stock_data.index, stock_data['MA_5'], label='5일 이평선', color='red', alpha=0.7)
        ax1.plot(stock_data.index, stock_data['MA_20'], label='20일 이평선', color='blue', alpha=0.7)
        ax1.plot(stock_data.index, stock_data['MA_50'], label='50일 이평선', color='green', alpha=0.7)
        
        ax1.fill_between(stock_data.index, stock_data['BB_Lower'], stock_data['BB_Upper'], 
                        alpha=0.2, color='gray', label='볼린저 밴드')
        
        ax1.set_title('주가와 이동평균선, 볼린저 밴드')
        ax1.set_ylabel('가격 (원)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[1]
        ax2.plot(stock_data.index, stock_data['RSI'], color='purple', linewidth=2)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='과매수선 (70)')
        ax2.axhline(y=30, color='blue', linestyle='--', alpha=0.7, label='과매도선 (30)')
        ax2.fill_between(stock_data.index, 70, 100, alpha=0.2, color='red')
        ax2.fill_between(stock_data.index, 0, 30, alpha=0.2, color='blue')
        
        ax2.set_title('RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axes[2]
        ax3.plot(stock_data.index, stock_data['MACD'], label='MACD', color='blue', linewidth=2)
        ax3.plot(stock_data.index, stock_data['MACD_Signal'], label='Signal', color='red', linewidth=2)
        ax3.bar(stock_data.index, stock_data['MACD_Histogram'], label='Histogram', 
               color='gray', alpha=0.6, width=1)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_title('MACD')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 거래량과 가격
        ax4 = axes[3]
        ax4_twin = ax4.twinx()
        
        # 거래량 (막대그래프)
        colors = ['red' if close < open_price else 'blue' 
                 for close, open_price in zip(stock_data['Close'], stock_data['Open'])]
        ax4.bar(stock_data.index, stock_data['Volume'], color=colors, alpha=0.6, width=1)
        ax4.plot(stock_data.index, stock_data['Volume_MA'], color='orange', linewidth=2, label='거래량 이평선')
        
        # 가격 (선그래프)
        ax4_twin.plot(stock_data.index, stock_data['Close'], color='black', linewidth=2, label='종가')
        
        ax4.set_title('거래량과 주가')
        ax4.set_xlabel('날짜')
        ax4.set_ylabel('거래량')
        ax4_twin.set_ylabel('가격 (원)')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 기술적 지표 차트 저장: {save_path}")
        
        plt.close()
    
    def plot_correlation_heatmap(self, data, save_path=None):
        """
        상관관계 히트맵을 그립니다.
        
        Args:
            data (pd.DataFrame): 분석 데이터
            save_path (str): 저장 경로
        """
        # 주요 지표들만 선택
        key_indicators = ['Close', 'Volume', 'Daily_Return', 'RSI', 'MACD', 
                         'BB_Position', 'Volatility', 'MA_20', 'MA_50']
        
        available_indicators = [col for col in key_indicators if col in data.columns]
        correlation_data = data[available_indicators].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        
        sns.heatmap(correlation_data, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.2f')
        
        plt.title('주요 지표 간 상관관계', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 상관관계 히트맵 저장: {save_path}")
        
        plt.close()
    
    def plot_performance_comparison(self, performance_data, save_path=None):
        """
        종목별 성과 비교 차트를 그립니다.
        
        Args:
            performance_data (pd.DataFrame): 성과 분석 데이터
            save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('종목별 성과 비교', fontsize=16, fontweight='bold')
        
        # 1. 수익률 vs 변동성 산점도
        ax1 = axes[0, 0]
        scatter = ax1.scatter(performance_data['연환산변동성(%)'], 
                            performance_data['연환산수익률(%)'],
                            c=performance_data['샤프비율'],
                            cmap='RdYlGn',
                            s=100,
                            alpha=0.7)
        
        # 종목명 표시
        for i, row in performance_data.iterrows():
            ax1.annotate(row['Name'], 
                        (row['연환산변동성(%)'], row['연환산수익률(%)']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        ax1.set_xlabel('연환산 변동성 (%)')
        ax1.set_ylabel('연환산 수익률 (%)')
        ax1.set_title('위험-수익률 분포 (색상: 샤프비율)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='샤프비율')
        
        # 2. 총 수익률 순위
        ax2 = axes[0, 1]
        sorted_data = performance_data.sort_values('총수익률(%)', ascending=True)
        colors = ['red' if x < 0 else 'blue' for x in sorted_data['총수익률(%)']]
        
        bars = ax2.barh(range(len(sorted_data)), sorted_data['총수익률(%)'], color=colors, alpha=0.7)
        ax2.set_yticks(range(len(sorted_data)))
        ax2.set_yticklabels(sorted_data['Name'])
        ax2.set_xlabel('총 수익률 (%)')
        ax2.set_title('종목별 총 수익률')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 값 표시
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + (1 if width >= 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left' if width >= 0 else 'right', va='center')
        
        # 3. 샤프 비율 비교
        ax3 = axes[1, 0]
        sorted_sharpe = performance_data.sort_values('샤프비율', ascending=True)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_sharpe)))
        
        bars = ax3.barh(range(len(sorted_sharpe)), sorted_sharpe['샤프비율'], color=colors, alpha=0.8)
        ax3.set_yticks(range(len(sorted_sharpe)))
        ax3.set_yticklabels(sorted_sharpe['Name'])
        ax3.set_xlabel('샤프 비율')
        ax3.set_title('종목별 샤프 비율')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. 최대 낙폭 비교
        ax4 = axes[1, 1]
        sorted_dd = performance_data.sort_values('최대낙폭(%)', ascending=False)  # 낙폭이 클수록 위험
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(sorted_dd)))
        
        bars = ax4.barh(range(len(sorted_dd)), sorted_dd['최대낙폭(%)'], color=colors, alpha=0.8)
        ax4.set_yticks(range(len(sorted_dd)))
        ax4.set_yticklabels(sorted_dd['Name'])
        ax4.set_xlabel('최대 낙폭 (%)')
        ax4.set_title('종목별 최대 낙폭')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 성과 비교 차트 저장: {save_path}")
        
        plt.close()
    
    def plot_time_analysis(self, weekday_data, monthly_data, save_path=None):
        """
        시간대별 분석 차트를 그립니다.
        
        Args:
            weekday_data (pd.DataFrame): 요일별 분석 데이터
            monthly_data (pd.DataFrame): 월별 분석 데이터
            save_path (str): 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('시간대별 수익률 분석', fontsize=16, fontweight='bold')
        
        # 1. 요일별 평균 수익률
        ax1 = axes[0, 0]
        weekday_avg = weekday_data.groupby('DayName')['평균수익률'].mean()
        weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일']
        weekday_avg = weekday_avg.reindex(weekday_order)
        
        colors = ['red' if x < 0 else 'blue' for x in weekday_avg.values]
        bars = ax1.bar(weekday_avg.index, weekday_avg.values, color=colors, alpha=0.7)
        ax1.set_title('요일별 평균 수익률')
        ax1.set_ylabel('평균 수익률 (%)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{height:.3f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        # 2. 요일별 양의 수익률 비율
        ax2 = axes[0, 1]
        win_rate_avg = weekday_data.groupby('DayName')['양의수익률비율'].mean()
        win_rate_avg = win_rate_avg.reindex(weekday_order)
        
        bars = ax2.bar(win_rate_avg.index, win_rate_avg.values, 
                      color=plt.cm.RdYlGn(win_rate_avg.values/100), alpha=0.8)
        ax2.set_title('요일별 상승 확률')
        ax2.set_ylabel('상승 확률 (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 50% 기준선
        ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% 기준선')
        ax2.legend()
        
        # 3. 월별 수익률 트렌드 (히트맵)
        if 'YearMonth' in monthly_data.columns:
            ax3 = axes[1, 0]
            
            # 월별 데이터 피벗
            monthly_pivot = monthly_data.pivot_table(
                values='mean', 
                index='Symbol', 
                columns='YearMonth', 
                aggfunc='mean'
            )
            
            sns.heatmap(monthly_pivot * 100, 
                       cmap='RdYlBu_r', 
                       center=0, 
                       annot=False,
                       cbar_kws={'label': '월별 평균 수익률 (%)'},
                       ax=ax3)
            
            ax3.set_title('종목별 월별 수익률 히트맵')
            ax3.set_xlabel('월')
            ax3.set_ylabel('종목')
        
        # 4. 변동성 월별 트렌드
        ax4 = axes[1, 1]
        if len(monthly_data) > 0:
            monthly_vol = monthly_data.groupby('YearMonth')['std'].mean() * 100
            ax4.plot(range(len(monthly_vol)), monthly_vol.values, 
                    marker='o', linewidth=2, markersize=6, color='orange')
            ax4.set_title('월별 평균 변동성 추이')
            ax4.set_xlabel('월')
            ax4.set_ylabel('변동성 (%)')
            ax4.grid(True, alpha=0.3)
            
            # x축 레이블 설정
            ax4.set_xticks(range(0, len(monthly_vol), max(1, len(monthly_vol)//6)))
            ax4.set_xticklabels([str(monthly_vol.index[i])[:7] for i in range(0, len(monthly_vol), max(1, len(monthly_vol)//6))], 
                               rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 시간대별 분석 차트 저장: {save_path}")
        
        plt.close()
    
    def create_interactive_dashboard(self, data, save_path=None):
        """
        인터랙티브 대시보드를 생성합니다.
        
        Args:
            data (pd.DataFrame): 주식 데이터
            save_path (str): 저장 경로
        """
        # 주요 종목들 선택
        symbols = data['Symbol'].unique()[:5]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('주가 추이', '정규화된 가격 비교', '거래량', 'RSI', 'MACD', '일일 수익률 분포'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, symbol in enumerate(symbols):
            stock_data = data[data['Symbol'] == symbol]
            company_name = stock_data['Name'].iloc[0]
            color = colors[i % len(colors)]
            
            # 1. 주가 추이
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data['Close'],
                          name=f'{company_name}',
                          line=dict(color=color, width=2),
                          showlegend=True),
                row=1, col=1
            )
            
            # 2. 정규화된 가격
            normalized_price = (stock_data['Close'] / stock_data['Close'].iloc[0]) * 100
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=normalized_price,
                          name=f'{company_name} (정규화)',
                          line=dict(color=color, width=2, dash='dot'),
                          showlegend=False),
                row=1, col=2
            )
            
            # 3. 거래량 (첫 번째 종목만)
            if i == 0:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['Volume'],
                              name='거래량',
                              line=dict(color='blue', width=1),
                              showlegend=False),
                    row=2, col=1
                )
            
            # 4. RSI (첫 번째 종목만)
            if i == 0:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['RSI'],
                              name='RSI',
                              line=dict(color='purple', width=2),
                              showlegend=False),
                    row=2, col=2
                )
                
                # RSI 기준선
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="과매수", row=2, col=2)
                fig.add_hline(y=30, line_dash="dash", line_color="blue", 
                             annotation_text="과매도", row=2, col=2)
            
            # 5. MACD (첫 번째 종목만)
            if i == 0:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['MACD'],
                              name='MACD',
                              line=dict(color='blue', width=2),
                              showlegend=False),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'],
                              name='Signal',
                              line=dict(color='red', width=2),
                              showlegend=False),
                    row=3, col=1
                )
            
            # 6. 일일 수익률 히스토그램 (모든 종목)
            fig.add_trace(
                go.Histogram(x=stock_data['Daily_Return'] * 100,
                           name=f'{company_name} 수익률',
                           opacity=0.7,
                           nbinsx=30),
                row=3, col=2
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text="주식 데이터 인터랙티브 대시보드",
                x=0.5,
                font=dict(size=20)
            ),
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        # 축 레이블 설정
        fig.update_xaxes(title_text="날짜", row=1, col=1)
        fig.update_xaxes(title_text="날짜", row=1, col=2)
        fig.update_xaxes(title_text="날짜", row=2, col=1)
        fig.update_xaxes(title_text="날짜", row=2, col=2)
        fig.update_xaxes(title_text="날짜", row=3, col=1)
        fig.update_xaxes(title_text="일일 수익률 (%)", row=3, col=2)
        
        fig.update_yaxes(title_text="가격 (원)", row=1, col=1)
        fig.update_yaxes(title_text="정규화된 가격", row=1, col=2)
        fig.update_yaxes(title_text="거래량", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=2)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="빈도", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"💾 인터랙티브 대시보드 저장: {save_path}")

def main():
    """메인 실행 함수"""
    viz = StockVisualization()
    
    print("📊 주식 데이터 시각화 시작")
    print("="*50)
    
    # 전처리된 데이터 로드
    processed_files = [f for f in os.listdir('data/processed/') if f.startswith('processed_stock_data_')]
    if not processed_files:
        print("❌ 시각화할 데이터 파일이 없습니다.")
        return
    
    latest_file = max(processed_files)
    data_path = f"data/processed/{latest_file}"
    data = viz.load_data(data_path)
    
    if data is None:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 주가 추이 차트
    print("📈 주가 추이 차트 생성 중...")
    viz.plot_stock_prices(
        data, 
        save_path=f"results/plots/stock_prices_{timestamp}.png"
    )
    
    # 2. 기술적 지표 차트 (첫 번째 종목)
    first_symbol = data['Symbol'].iloc[0]
    print(f"📊 {first_symbol} 기술적 지표 차트 생성 중...")
    viz.plot_technical_indicators(
        data, 
        first_symbol,
        save_path=f"results/plots/technical_indicators_{timestamp}.png"
    )
    
    # 3. 상관관계 히트맵
    print("🔥 상관관계 히트맵 생성 중...")
    viz.plot_correlation_heatmap(
        data,
        save_path=f"results/plots/correlation_heatmap_{timestamp}.png"
    )
    
    # 4. 성과 비교 차트 (분석 결과 파일이 있는 경우)
    analysis_files = [f for f in os.listdir('results/analysis/') if 'performance_metrics' in f]
    if analysis_files:
        latest_analysis = max(analysis_files)
        performance_data = pd.read_csv(f"results/analysis/{latest_analysis}", encoding='utf-8-sig')
        
        print("🏆 성과 비교 차트 생성 중...")
        viz.plot_performance_comparison(
            performance_data,
            save_path=f"results/plots/performance_comparison_{timestamp}.png"
        )
    
    # 5. 시간대별 분석 차트
    weekday_files = [f for f in os.listdir('results/analysis/') if 'weekday_analysis' in f]
    monthly_files = [f for f in os.listdir('results/analysis/') if 'monthly_analysis' in f]
    
    if weekday_files and monthly_files:
        weekday_data = pd.read_csv(f"results/analysis/{max(weekday_files)}", encoding='utf-8-sig')
        monthly_data = pd.read_csv(f"results/analysis/{max(monthly_files)}", encoding='utf-8-sig')
        
        print("📅 시간대별 분석 차트 생성 중...")
        viz.plot_time_analysis(
            weekday_data,
            monthly_data,
            save_path=f"results/plots/time_analysis_{timestamp}.png"
        )
    
    # 6. 인터랙티브 대시보드
    print("🖥️ 인터랙티브 대시보드 생성 중...")
    viz.create_interactive_dashboard(
        data,
        save_path=f"results/plots/interactive_dashboard_{timestamp}.html"
    )
    
    print("\n✅ 모든 시각화 완료!")
    print(f"📁 결과 파일들이 'results/plots/' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 