# Universal 3D Gaussian Splatting Pipeline

### 프로젝트 설명
이 파이프라인은 **이미지 전처리**부터 **Gaussian Splatting** 및 **물리 충돌 메쉬 생성**까지 포함한 고성능 3D 모델링 알고리즘입니다.

### 실행 방법
1. Python 의존성 설치:
   ```bash
   pip install -r requirements.txt

   COLMAP 설치:

bash
코드 복사
sudo apt install colmap

data/raw_images/에 입력 이미지 추가 후 실행:

bash
코드 복사
python run_pipeline.py


---

### **5. 업로드 및 실행**
1. 디렉토리 정리 후 GitHub에 업로드:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Gaussian Splatting pipeline"
   git branch -M main
   git remote add origin <your-repo-link>
   git push -u origin main
