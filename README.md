# CNC-Net_PyTorch

This repository contains the official code to reproduce the results from the paper:

**𝐂𝐍𝐂-𝐍𝐞𝐭: 𝐒𝐞𝐥𝐟-𝐒𝐮𝐩𝐞𝐫𝐯𝐢𝐬𝐞𝐝 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐟𝐨𝐫 𝐂𝐍𝐂 𝐌𝐚𝐜𝐡𝐢𝐧𝐢𝐧𝐠 𝐎𝐩𝐞𝐫𝐚𝐭𝐢𝐨𝐧𝐬**

\[[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yavartanoo_CNC-Net_Self-Supervised_Learning_for_CNC_Machining_Operations_CVPR_2024_paper.html)\]\[[arXiv](https://arxiv.org/abs/2312.09925)\] \[[YouTube](https://www.youtube.com/watch?v=0wg5aV-q7XU&t=1s)\]


<p align="center">
<img src="source/Framework.png" width="100%"/>  
</p>

## Installation (Google Cloud TPU)

### Prerequisites
* A Google Cloud TPU VM (v2 or later)
* Conda (Miniconda or Anaconda)

### 1. Clone the repository
```bash
git clone https://github.com/myavartanoo/CNC-Net_PyTorch.git
cd CNC-Net_PyTorch
```

### 2. Create the conda environment
```bash
conda env create -f environment.yml
conda activate cnctorch-tpu
```

Or install from `requirements.txt` into an existing Python >= 3.9 environment:
```bash
pip install -r requirements.txt
```

### Dependencies
* Python >= 3.9
* PyTorch >= 2.1
* torch_xla >= 2.1
* numpy
* open3d
* pysdf
* scipy
* trimesh
* h5py
* PyMCubes
* pymesh2
* plyfile
* tqdm

## Quick Start
To run the model for a given object on TPU:
```bash
python run.py --input_object {object_name} --experiment {experiment_name}
```
For example:
```bash
python run.py --input_object '0.off' --experiment exp_0
```


### Citation
If you find our code or paper useful, please consider citing:
```
@inproceedings{CNC-Net,
  title={CNC-Net: Self-Supervised Learning for CNC Machining Operations},
  author={Mohsen Yavatanoo and Sangmin Hong and Reyhaneh Neshatavar and Kyoung Mu Lee},
  booktitle={The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}
```

