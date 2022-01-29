# conda create a new environment
# make sure system cuda version is the same with pytorch cuda
# # follow the instruction of Pyotrch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# export PATH=/usr/local/cuda-11.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# conda create --name metric_learning 
# activate this enviroment
# conda activate metric_learning 

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

test if pytorch is installed successfully
python -c "import torch; print(torch.__version__)"
nvcc --version # should be same with that of torch_version_cuda (they should be the same)
python -c "import torch; print(torch.version.cuda)"

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu111.html

pip install tqdm
pip install ogb

### check the version of ogb installed, if it is not the latest
python -c "import ogb; print(ogb.__version__)"
# please update the version by running
pip install -U ogb

conda install -c conda-forge rdkit

pip install h5py