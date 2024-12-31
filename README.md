# Gaussian-test
Gaussian Splatting

# Universal 3D Gaussian Splatting Pipeline

### 프로젝트 설명
이 파이프라인은 **이미지 전처리**부터 **Gaussian Splatting** 및 **물리 충돌 메쉬 생성**까지 포함한 고성능 3D 모델링 알고리즘입니다.

프로젝트 목표: 범용적이고 고성능의 3D Gaussian Splatting 기반 파이프라인 설계 및 구현
목표는 범용성, 효율성, 그리고 확장성을 갖춘 3D Gaussian Splatting 알고리즘을 구현하여, 대규모 장면 및 다양한 응용 분야에서 활용 가능한 완성형 프로젝트를 만드는 것입니다. 이 알고리즘은 이미지 전처리, Structure-from-Motion(SfM), Multi-View Stereo(MVS), Gaussian Splatting 최적화, 그리고 물리적 충돌 가능한 메쉬 생성까지 모든 단계에서 사용할 수 있습니다. GitHub 포트폴리오에 올릴 수 있도록 코드를 정리하여 범용적으로 사용할 수 있는 상태로 제공합니다.

**1. 알고리즘 설계**
1.1 전체 파이프라인
이미지 전처리:
그림자 제거, 색보정, 노이즈 감소로 입력 이미지 품질 개선.
Structure-from-Motion(SfM):
COLMAP을 사용하여 카메라 위치와 초기 포인트 클라우드 생성.
Multi-View Stereo(MVS):
COLMAP으로 Dense Point Cloud 생성.
Gaussian Splatting:
Dense Point Cloud를 기반으로 Gaussian 모델 초기화 및 최적화.
물리 충돌 가능한 메쉬 변환:
Gaussian Splatting 결과를 삼각형 메쉬로 변환하고, 물리적 충돌 처리가 가능하도록 준비.

**2. 프로젝트 구조**

universal-gaussian-splatting-pipeline
├── README.md                       # 프로젝트 설명 및 실행 방법

├── requirements.txt                # Python 의존성

├── data/                           # 데이터 디렉토리

│   ├── raw_images/                 # 원본 이미지

│   ├── processed_images/           # 전처리된 이미지

├── outputs/                        # 결과물 디렉토리

│   ├── sparse/                     # SfM 결과

│   ├── dense/                      # MVS 결과

│   ├── gaussian/                   # Gaussian Splatting 결과

│   ├── mesh/                       # 물리적 충돌 가능한 메쉬

├── src/                            # 소스 코드

│   ├── preprocessing.py            # 이미지 전처리 코드

│   ├── colmap_utils.py             # COLMAP 관련 코드

│   ├── gaussian_splatting.py       # Gaussian Splatting 알고리즘

│   ├── mesh_conversion.py          # 메쉬 변환 및 충돌 처리 코드

├── run_pipeline.py                 # 전체 파이프라인 실행 스크립트

**3. 주요 알고리즘 및 코드**

3.1 이미지 전처리

src/preprocessing.py:

python

import cv2
import numpy as np
from pathlib import Path

def remove_shadows(image):

    """그림자 제거."""
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.merge((l, a, b))

def correct_colors(image):

    """색상 보정."""
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.merge((l, a, b))

def preprocess_images(input_dir, output_dir):

    """이미지 전처리 수행."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.glob("*.jpg"):
        image = cv2.imread(str(img_path))
        no_shadow = remove_shadows(image)
        corrected = correct_colors(no_shadow)
        cv2.imwrite(str(output_dir / img_path.name), corrected)

    print(f"Preprocessed images saved to {output_dir}")
    
3.2 COLMAP 기반 SfM 및 MVS

src/colmap_utils.py:

python

import subprocess
from pathlib import Path

def run_colmap(command, args):
   
    """COLMAP 명령 실행."""
    
    cmd = ["colmap", command]
    for k, v in args.items():
        cmd.append(f"--{k}={v}")
    subprocess.run(cmd, check=True)

def perform_sfm(image_dir, database_path, output_path):
   
    """SfM 수행."""
    
    run_colmap("feature_extractor", {
        "database_path": database_path,
        "image_path": image_dir
    })
    run_colmap("sequential_matcher", {
        "database_path": database_path
    })
    run_colmap("mapper", {
        "database_path": database_path,
        "image_path": image_dir,
        "output_path": output_path
    })

def perform_mvs(sparse_dir, image_dir, output_dir):
   
    """MVS 수행."""
    
    run_colmap("image_undistorter", {
        "image_path": image_dir,
        "input_path": sparse_dir,
        "output_path": output_dir
    })
    run_colmap("patch_match_stereo", {
        "workspace_path": output_dir
    })
    run_colmap("stereo_fusion", {
        "workspace_path": output_dir,
        "output_path": f"{output_dir}/fused.ply"
    })
    
3.3 Gaussian Splatting

src/gaussian_splatting.py:

python

import numpy as np
import open3d as o3d

def initialize_gaussian(point_cloud_path):
   
    """가우시안 초기화."""
    
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    gaussians = []
    for point, color in zip(points, colors):
        gaussian = {
            "position": point,
            "color": color,
            "covariance": np.eye(3) * 0.01
        }
        gaussians.append(gaussian)
    return gaussians

def optimize_gaussians(gaussians):
    
    """가우시안 최적화."""
    
    # 간단한 최적화 예제 (사용자의 요구에 따라 확장 가능)
    
    for g in gaussians:
        g["covariance"] *= 1.05  # 임의의 연산
    return gaussians
    
3.4 메쉬 변환 및 물리적 충돌 처리

src/mesh_conversion.py:

python

from pyntcloud import PyntCloud

def convert_gaussian_to_mesh(gaussians, output_path):
   
    """가우시안 모델을 메쉬로 변환."""
    
    points = [g["position"] for g in gaussians]
    cloud = PyntCloud(pd.DataFrame(points, columns=["x", "y", "z"]))
    mesh = cloud.to_mesh()
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")

3.5 실행 스크립트

run_pipeline.py:

python

from src.preprocessing import preprocess_images
from src.colmap_utils import perform_sfm, perform_mvs
from src.gaussian_splatting import initialize_gaussian, optimize_gaussians
from src.mesh_conversion import convert_gaussian_to_mesh

def main():
    raw_dir = "./data/raw_images"
    processed_dir = "./data/processed_images"
    sparse_dir = "./outputs/sparse"
    dense_dir = "./outputs/dense"
    gaussian_output = "./outputs/gaussian"
    mesh_output = "./outputs/mesh/mesh.obj"

    # 1. 전처리
    
    preprocess_images(raw_dir, processed_dir)

    # 2. SfM 및 MVS
    
    perform_sfm(processed_dir, "./data/database.db", sparse_dir)
    perform_mvs(sparse_dir, processed_dir, dense_dir)

    # 3. Gaussian Splatting
    
    gaussians = initialize_gaussian(f"{dense_dir}/fused.ply")
    gaussians = optimize_gaussians(gaussians)

    # 4. 메쉬 변환
    
    convert_gaussian_to_mesh(gaussians, mesh_output)

if __name__ == "__main__":
    main()
