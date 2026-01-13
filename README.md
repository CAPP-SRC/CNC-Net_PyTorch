# CNC-Net_PyTorch - CPU-only

This repository contains edited code to reproduce **CPU-only** the results from the following paper:

**𝐂𝐍𝐂-𝐍𝐞𝐭: 𝐒𝐞𝐥𝐟-𝐒𝐮𝐩𝐞𝐫𝐯𝐢𝐬𝐞𝐝 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠 𝐟𝐨𝐫 𝐂𝐍𝐂 𝐌𝐚𝐜𝐡𝐢𝐧𝐢𝐧𝐠 𝐎𝐩𝐞𝐫𝐚𝐭𝐢𝐨𝐧𝐬**

\[[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yavartanoo_CNC-Net_Self-Supervised_Learning_for_CNC_Machining_Operations_CVPR_2024_paper.html)\]\[[arXiv](https://arxiv.org/abs/2312.09925)\] \[[YouTube](https://www.youtube.com/watch?v=0wg5aV-q7XU&t=1s)\]


<p align="center">
<img src="source/Framework.png" width="100%"/>  
</p>

## Installation
Clone this repository into any place you want.
```
git clone https://github.com/CAPP-SRC/CNC-Net_PyTorch.git
cd CNC-Net_PyTorch
```

### Dependencies / Installation [WIP]
Install the dependencies:

* Python 3.8.18

* pytorch-cpu 1.12.1
* conda install -c pytorch pytorch=1.12.1 torchvision cpuonly

* pip install numpy==1.24.4

* pip install open3d-cpu

* pip install fvcore iopath
* conda install -c fvcore -c iopath -c conda-forge fvcore iopath
* sudo apt install build-essential
* pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

* pip install scipy
* pip install PyMCubes==0.1.4

* conda install plyfile
* pip install plyfile

* pip install --force-reinstall charset-normalizer==3.1.0

* pip install required/pymesh2-0.3-cp38-cp38-linux_x86_64.whl

* pip install pysdf

## Quick Start
To run the model **in CPU-only environments** for a given object follow the below command.
```
python run.py --input_object {objec_name} --experiment {experiment_name}
```
For example:
```
python run.py --input_object '0.off' --experiment exp_0
```


### Citation
If you find the code and/or paper useful, please consider citing:
```
@inproceedings{CNC-Net,
  title={CNC-Net: Self-Supervised Learning for CNC Machining Operations},
  author={Mohsen Yavatanoo and Sangmin Hong and Reyhaneh Neshatavar and Kyoung Mu Lee},
  booktitle={The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}
```

