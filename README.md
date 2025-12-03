# 열처리 공정 품질 예측 시스템

제조 공정 데이터 분석 및 시계열 예측 모델을 활용한 불량률 예측 시스템

## 프로젝트 구조

```
src/
├── data/                    # 데이터 파일
├── models/                  # 모델 정의 (DLinear, NLinear, LSTM)
├── onnx/                    # 학습된 ONNX 모델
├── analyze.ipynb            # 1단계: 통계 분석
├── predict.ipynb            # 2단계: 시계열 예측 모델 학습
├── dashboard.ipynb          # 3단계: 실시간 모니터링 대시보드
├── requirements.txt         # 필요 패키지
└── result.csv               # 모델 성능 결과
```

## 환경 설정

```bash
pip install -r src/requirements.txt
```

**주요 패키지:**
- Python 3.8+
- pandas, numpy, scipy
- scikit-learn, torch
- onnxruntime
- matplotlib, flask

## 실행 순서

### 1. 통계 분석 (analyze.ipynb)
```bash
jupyter notebook src/analyze.ipynb
```
- 37개 공정 변수 통계 분석
- 핵심 변수 3개 선정 (Cohen's d, t-test, Mann-Whitney U)
- 출력: `src/data/analyze.csv`

### 2. 예측 모델 학습 (predict.ipynb)
```bash
jupyter notebook src/predict.ipynb
```
- 시계열 예측 모델 학습 (DLinear, NLinear, LSTM)
- 36개 조합 실험 (3 모델 × 3 타겟 × 2 학습률 × 2 손실함수)
- 출력: `src/onnx/*.onnx`, `src/result.csv`

### 3. 모니터링 대시보드 (dashboard.ipynb)
```bash
jupyter notebook src/dashboard.ipynb
```
- Flask 기반 실시간 모니터링 시스템
- 10분 후 온도 예측 시각화


## 주요 결과

### 통계 분석 결과
- 핵심 변수: 소입로 온도 1 Zone_최저, 건조로 온도 2 Zone_최저, 소입로 온도 4 Zone_최저
- Cohen's d: 1.68~2.31 (큰 효과 크기)

### 최적 모델 성능
| 타겟 변수 | 모델 | MAE | RMSE | SMAPE(%) |
|----------|------|-----|------|----------|
| 건조로 온도 2 Zone | LSTM | 0.334 | 0.563 | 98.15 |
| 소입로 온도 1 Zone | LSTM | 0.325 | 0.669 | 74.93 |
| 소입로 온도 4 Zone | LSTM | 0.312 | 0.456 | 115.97 |

**최적 하이퍼파라미터:** 학습률 0.01 + L1Loss
