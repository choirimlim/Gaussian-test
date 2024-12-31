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
