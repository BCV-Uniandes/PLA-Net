conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# test if pytorch is installed successfully
python -c "import torch; print(torch.__version__)"
nvcc --version # should be same with that of torch_version_cuda (they should be the same)
python -c "import torch; print(torch.version.cuda)"

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

pip install tqdm
pip install ogb

### check the version of ogb installed, if it is not the latest
python -c "import ogb; print(ogb.__version__)"
# please update the version by running
pip install -U ogb

conda install -c conda-forge rdkit

pip install h5py