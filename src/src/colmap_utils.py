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
