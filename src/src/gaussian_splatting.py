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
