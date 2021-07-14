import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data

import time
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from dotenv import dotenv_values
import open3d as o3d


shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


# An Tao's code from the poincloud repos
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', class_choice=None,
            num_points=2048, split='train', load_name=True, load_file=True,
            segmentation=False, random_rotate=False, random_jitter=False, 
            random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetpart'] and segmentation == True:
            raise AssertionError

        self.root = os.path.join(root, dataset_name + '_' + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)    # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = self.load_json(self.path_file_all)    # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                id_choice = shapenetpart_cat2id[class_choice]
                self.seg_num_all = shapenetpart_seg_num[id_choice]
                self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json'%type)
            self.path_file_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label, seg, name, file
        else:
            return point_set, label, name, file

    def __len__(self):
        return self.data.shape[0]
###

def custom_draw_geometry_with_rotation(pcd):
    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.light_on = True
        
        return False

    o3d.visualization.draw_geometries_with_animation_callback(pcd, change_background_to_black)

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_views", type=int, help="batch size", default=3)
    parser.add_argument("--resolution", type=int, help="number of epochs", default=20)
    parser.add_argument("--num_points", type=int, help="number of epochs", default=1024)
    parser.add_argument("--use_checkpoint", type=str, help="specify the checkpoint root", default="")
    parser.add_argument("--index", type=int, help="batch size", required=True)
    parser.add_argument("--split", help="batch size", required=True)

    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    num_views = args.num_views
    resolution = args.resolution
    num_points = args.num_points

    env_vars = dotenv_values('../.env')
    dataset_name = 'shapenetcorev2'
    split = args.split
    d = Dataset(root=env_vars["SHAPENET_DATASET_PATH"], dataset_name=dataset_name, num_points=args.num_points, split=split)
    

    if not os.path.exists(env_vars["SHAPENET_DATASET_PATH"] + "/ShapeNetPC"):
        os.mkdir(env_vars["SHAPENET_DATASET_PATH"] + "/ShapeNetPC")


    if (args.index % 100) == 0:
        print(f'Processing renderings {args.index} / {d.__len__()}')


    k = 0
    ps, lb, n, f = d.__getitem__(args.index)
    renderer = o3d.visualization.rendering.OffscreenRenderer(137, 137)
    mat = rendering.Material()
    mat.shader = 'defaultLit'
    # renderer.scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.HARD_SHADOWS, (0.577, -0.577, -0.577))
    renderer.scene.camera.look_at([0, 0.25, -.5], [0, 1, 1], [0,1,0])

    label, id = f[:-4].split('/')
    dir_path = env_vars["SHAPENET_DATASET_PATH"] + f'/ShapeNetPC_incomplete/{label}/{id}/rendering'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    k += 1
    ps = ps.numpy()
    ps = ps # scale down pointcloud

    # Cut points to create incomplete pc
    axis = np.random.randint(3)
    ps = ps[np.argsort(ps[:, axis])]
    ps = ps[:600, :]

    for i in range(num_views):
        k += 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ps)
        
        # Add rotation
        alpha = 0
        beta = 2 * np.pi * (i / num_views)  
        eps = 0

        # r_x = np.array([[1, 0, 0],
        #             [0, np.cos(alpha), -np.sin(alpha)],
        #             [0, np.sin(alpha), np.cos(alpha)]])

        r_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])

        # r_z = np.array([[np.cos(eps), -np.sin(eps), 0],
        #             [np.sin(eps), np.cos(eps), 0],
        #             [0, 0, 1]])


        pcd = pcd.rotate(r_y)
        pcd = pcd.translate(np.array([0,0,-1]))
        pc = np.asarray(pcd.points)

        points = []
        for j in pc:
            m = o3d.geometry.TriangleMesh.create_sphere(radius=.035, resolution=resolution).translate(j)
            points.append(m)

        k = 0
        for j in points:
            renderer.scene.add_geometry(f'mesh_{k}', j, mat)
            k += 1

        img = renderer.render_to_image()
        if i < 10:
            o3d.io.write_image(dir_path + f'/0{i}.png', img)
        else:
            o3d.io.write_image(dir_path + f'/{i}.png', img)

        renderer.scene.clear_geometry()