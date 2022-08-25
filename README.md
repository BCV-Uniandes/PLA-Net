# PLA-Net: Modeling Protein-Ligand Interactions with Graph Convolutional Networks for Interpretable Pharmaceutical Discovery

Paola Ruiz Puentes, Laura Rueda-Gensini, Natalia Valderrama, Isabela Hernández, Cristina González, Laura Daza, Carolina Muñoz-Camargo, Juan C. Cruz, Pablo Arbeláez

This repository contains the official implementation of PLA-Net, submitted for revision to *Scientific Reports*. 

## Installation
The following steps are required in order to run PLA-Net:<br />


export PATH=/usr/local/cuda-11.0/bin:$PATH <br />
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH <br />


conda create --name PLA-Net <br />
conda activate PLA-Net <br />


Run env.sh

## Models
We provide trained models available for download in the following [link](http://157.253.243.19/PLA-Net/).

## Usage
To train each of the components of our method: LM, LM+Advs, LMPM and PLA-Net please refer to planet.sh file and run the desired models.

To evaluate each of the components of our method: LM, LM+Advs, LMPM and PLA-Net please run the corresponding bash file in the inference folder.
