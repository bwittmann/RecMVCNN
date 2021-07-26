"""Helper funtions."""

import numpy as np

from dotenv import dotenv_values
import open3d as o3d


env_vars = dotenv_values('.env')

def read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    Adapted from: ML3D Exercise 2
    """
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = [int(i) for i in fp.readline().strip().split(b' ')[1:]]
    translate = [float(i) for i in fp.readline().strip().split(b' ')[1:]]
    scale = [float(i) for i in fp.readline().strip().split(b' ')[1:]][0]
    line = fp.readline()
    return dims, translate, scale

def read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.
    Returns the model with accompanying metadata.
    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).
    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, _, _ = read_header(fp)
    raw_data = np.frombuffer(fp.read(), dtype=np.uint8)
    
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)
    data = data.reshape(dims)
    if fix_coords:
        data = np.transpose(data, (0, 2, 1))
    return data

def visualize_voxel_grid(voxel_grid): 
    """Visualizes voxel grid.
    voxel_grid: a np.array
    """
    voxels = []

    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[0]):
                if voxel_grid[i,j,k] >= 0.5:
                    voxel = [i, j, k]
                    voxels.append(voxel)

    pcd = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(np.array(voxels, dtype=int))
    pcd.points = points
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(points), 3)))

    o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1)
    o3d.visualization.draw_geometries([o3d_voxel_grid])

def save_voxel_grid(path, voxel_grid):
    """Save a voxel grid.
    path: absolute path of where to store the voxel grid
    voxel_grid: a np.array
    """
    voxels = []

    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[0]):
                if voxel_grid[i,j,k] >= 0.5:
                    voxel = [i, j, k]
                    voxels.append(voxel)


    pcd = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(np.array(voxels, dtype=int))
    pcd.points = points
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(points), 3)))
    o3d_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 1)
    o3d.io.write_voxel_grid(path, o3d_voxel_grid)


if __name__ == "__main__":
    voxel_grid = None
    
    with open(env_vars["SHAPENET_VOXEL_DATASET_PATH"] + "/02691156/1c26ecb4cd01759dc1006ed55bc1a3fc/model.binvox", "rb") as fptr:
            voxel_grid = read_as_3d_array(fptr).astype(np.float32)

    visualize_voxel_grid(voxel_grid)