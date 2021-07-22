# RecMVCNN

RecMVCNN is a lightweight multi-view based approach for classification and reconstruction of a 3D structure. It takes rendered images of a 3D structure like a mesh, a point cloud, or a voxel grid as input and simultaneous classifies the content of those rendered images and reconstructs the 3D shape in the form of a voxel grid. The architecture of our approach is depicted below.

<img src="docs/images/RecMVCNN_Architecture.png" alt="Architecture of our approach.">

The big advantage of this approach is that we can leverage the expressiveness of pre-trained convolutional backbones to extract features of those evenly spaced multi-view images. This allows our method to perform reasonably well without utilizing any explicit 3D input data.

For further details, ablation studies, and results, please refer to the report and presentation in the docs.

The table below shows multi-viee...


<div class="row">
<caption>Multi-view images of rendered point cloud.</caption>
  <div class="col-sm">
    <img src="docs/images/pc_00.png" width=120>
    <img src="docs/images/pc_01.png" width=120>
    <img src="docs/images/pc_02.png" width=120>
  </div>
<caption>Multi-view images of rendered incomplete point cloud.</caption>
  <div class="col-sm">
    <img src="docs/images/pc_inc_00.png" width=120>
    <img src="docs/images/pc_inc_01.png" width=120>
    <img src="docs/images/pc_inc_02.png" width=120>
  </div>
<caption>Multi-view images of rendered colored mesh.</caption>
  <div class="col-sm">
    <img src="docs/images/mesh_00.png" width=120>
    <img src="docs/images/mesh_01.png" width=120>
    <img src="docs/images/mesh_02.png" width=120>
  </div>
</div>



# Usage

## Data
The voxelized models and rendered images of corresponding meshes can be downloaded with the links bellow:
- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

If you want to train the model on renderings of point clouds, you should download the point cloud representations of ShapeNet objects from the repo below.
- ShapeNet point cloud representation for rendering: https://github.com/AnTao97/PointCloudDatasets

In addition, you have to render multi-view images of the point cloud representations yourself if desired. Place the folder containing the ShapeNet point cloud representation in the ShapeNet directory and run one of the scripts listed bellow. The second script renders incomplete point cloud representations. This is done by reducing the points in the point cloud to 60% with the help of a randomly selected plane used for splitting.

Complete point cloud representation: 

    bash generate_pointcloud_dataset.sh

Incomplete point cloud representation: 

    bash generate_pointcloud_incomplete_dataset.sh

## Environment Variables
To use this project, you must set several environment variables beforehand in a .env file using dotenv.

    PROJECT_DIR_PATH="Path/To/RecMVCNN"    
    SHAPENET_DATASET_PATH="Path/To/ShapeNet"
    SHAPENET_VOXEL_DATASET_PATH="Path/To/ShapeNet/ShapeNetVox32"
    SHAPENET_RENDERING_DATASET_PATH="Path/To/ShapeNet/ShapeNetRendering"
    SHAPENET_PC_RENDERING_DATASET_PATH="Path/To/ShapeNet/ShapeNetPC"
    SHAPENET_PC_INC_RENDERING_DATASET_PATH="Path/To/ShapeNet/ShapeNetPC_incomplete"


## Training

## Testing / Inference




todo:
get ridd of todos
improve generate_pointcloud_dataset.
incorporate incomplete pc as a flag
