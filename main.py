"""
주식 빅데이터 분석 프로젝트 메인 실행 파일
데이터 수집부터 예측까지 전체 파이프라인을 실행합니다.
"""

import os
import sys
import time
from datetime import datetime

# 프로젝트 모듈들 import
sys.path.append('src')

def print_header(title):
    """섹션 헤더를 출력합니다."""
    print("\n" + "="*80)
    print(f"🚀 {title}")
    print("="*80)

def print_step(step_num, description):
    """단계별 진행상황을 출력합니다."""
    print(f"\n📌 {step_num}단계: {description}")
    print("-" * 50)

def main():
    """메인 실행 함수"""
    start_time = time.time()
    
    print_header("한국 주식 시장 빅데이터 분석 프로젝트")
    print("🎯 프로젝트 목표: 주식 데이터 수집, 분석, 시각화 및 예측")
    print(f"🕒 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1단계: 데이터 수집
        print_step("1", "주식 데이터 수집")
        from data_collection import main as collect_data
        collect_data()
        
        # 잠시 대기
        time.sleep(2)
        
        # 2단계: 데이터 전처리
        print_step("2", "데이터 전처리 및 특성 엔지니어링")
        from data_preprocessing import main as preprocess_data
        preprocess_data()
        
        time.sleep(2)
        
        # 3단계: 데이터 분석
        print_step("3", "통계 분석 및 패턴 분석")
        from data_analysis import main as analyze_data
        analyze_data()
        
        time.sleep(2)
        
        # 4단계: 시각화
        print_step("4", "데이터 시각화 및 차트 생성")
        from visualization import main as visualize_data
        visualize_data()
        
        time.sleep(2)
        
        # 5단계: 예측 모델링
        print_step("5", "머신러닝 모델 훈련 및 예측")
        from prediction import main as predict_data
        predict_data()
        
        # 프로젝트 완료
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # 분 단위
        
        print_header("프로젝트 완료")
        print("✅ 모든 단계가 성공적으로 완료되었습니다!")
        print(f"⏱️ 총 실행 시간: {execution_time:.2f}분")
        print(f"🕒 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📂 생성된 결과 파일들:")
        print("  📁 data/raw/          - 원시 데이터")
        print("  📁 data/processed/    - 전처리된 데이터") 
        print("  📁 results/analysis/  - 분석 결과")
        print("  📁 results/plots/     - 시각화 차트")
        print("  📁 results/predictions/ - 예측 결과")
        print("  📁 results/models/    - 훈련된 모델")
        
        print("\n🎉 프로젝트가 성공적으로 완료되었습니다!")
        print("📊 interactive_dashboard.html 파일을 브라우저에서 열어 인터랙티브 대시보드를 확인하세요.")
        
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {e}")
        print("📝 필요한 라이브러리를 설치해주세요: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        print("🔍 오류 세부사항을 확인하고 다시 시도해주세요.")

def run_specific_step(step):
    """특정 단계만 실행합니다."""
    print_header(f"특정 단계 실행: {step}")
    
    try:
        if step == "collect" or step == "1":
            from data_collection import main as collect_data
            collect_data()
            
        elif step == "preprocess" or step == "2":
            from data_preprocessing import main as preprocess_data
            preprocess_data()
            
        elif step == "analyze" or step == "3":
            from data_analysis import main as analyze_data
            analyze_data()
            
        elif step == "visualize" or step == "4":
            from visualization import main as visualize_data
            visualize_data()
            
        elif step == "predict" or step == "5":
            from prediction import main as predict_data
            predict_data()
            
        else:
            print(f"❌ 알 수 없는 단계: {step}")
            print("사용 가능한 단계: collect(1), preprocess(2), analyze(3), visualize(4), predict(5)")
            
    except Exception as e:
        print(f"❌ {step} 단계 실행 중 오류: {e}")

if __name__ == "__main__":
    # 명령행 인수 확인
    if len(sys.argv) > 1:
        # 특정 단계 실행
        step = sys.argv[1].lower()
        run_specific_step(step)
    else:
        # 전체 파이프라인 실행
        main() 