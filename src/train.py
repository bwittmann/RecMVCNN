from Datasets import ShapeNetDataset

if __name__ == '__main__':
    '''Trial/test code'''
    from dotenv import dotenv_values
    from torch.utils.data import  DataLoader
    env_vars = dotenv_values('.env')
    dataset = ShapeNetDataset(env_vars['SHAPENET_VOXEL_DATASET_PATH'], env_vars['SHAPENET_RENDERING_DATASET_PATH'], 'data/ShapeNet.json')
    dataloader = DataLoader(dataset)
    shapenet_id, renderings, class_label, voxel = next(iter(dataloader))

