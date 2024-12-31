from pyntcloud import PyntCloud

def convert_gaussian_to_mesh(gaussians, output_path):
    """가우시안 모델을 메쉬로 변환."""
    points = [g["position"] for g in gaussians]
    cloud = PyntCloud(pd.DataFrame(points, columns=["x", "y", "z"]))
    mesh = cloud.to_mesh()
    mesh.export(output_path)
    print(f"Mesh saved to {output_path}")
