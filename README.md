# DVGFormer: Learning Camera Movement Control from Real-World Drone Videos


<a href="https://arxiv.org/abs/2412.09620"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
<a href="https://dvgformer.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" height=22.5></a>
<!-- <a href="https://huggingface.co/SPO-Diffusion-Models"><img src="https://img.shields.io/badge/Hugging-Face-yellow?style=for-the-badge" height=22.5></a> -->

<!-- [[Paper]()] &emsp; [[Project Page]()] &emsp;
<br> -->

Official implementation of our paper: 
<br>**Learning Camera Movement Control from Real-World Drone Videos**<br>
[**Yunzhong Hou**](https://hou-yz.github.io/), [**Liang Zheng**](https://zheng-lab-anu.github.io/), [**Philip Torr**](https://eng.ox.ac.uk/people/philip-torr/)<br>


## Installation
1. **Create and activate a Conda environment**:
    ```sh
    conda create -n dvg python=3.10
    conda activate dvg
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c conda-forge ffmpeg
    pip install -r requirements.txt
    ```

2. **Download evaluation data**
   
    For real city 3D scans from Google Earth, please download from this [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EhrvMtW9ow5KrpfPJlAnJ9wBjaaYqNEKx98NOXGFteJ3pg?e=d99AG4).

    For synthetic natural scenes, you can either generate your own version from the official git repo [princeton-vl/infinigen](https://github.com/princeton-vl/infinigen) or directly download from this [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EgQWiB64W6dCsuOko_UoNQoB9Zj4cb-SSlqLFdVZITJT7Q?e=MBvCGx). Note that our version has very basic graphic settings and you might need to generate your own version if you need higher graphics. 

    After downloading the evaluation environments, your folder should look like this


3. **Download training data**
   
    We provide the Colmap 3D reconstruction results and the filtered camera movement sequences in our DroneMotion-99k dataset. You can download either a minimal dataset with 10 videos and 129 sequences [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/ERIEM1bBgvVHtqgyN4T-7qoBmiHYaHcAdUUz5McREVuI_w?e=qwOBge) or the full dataset with 13,653 videos and 99,003 camera trajectories [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/EcHhl1KtZrdHn4wkDJ9Kcg4BtwQCP3f3hKUHS7PArhprnw?e=SRkFjl). 

    After downloading the training data, your folder should look like this


## Citation
Please cite our paper:
```
@article{hou2024dvgformer,
  author    = {Hou, Yunzhong and Zheng, Liang and Torr, Philip},
  title     = {Learning Camera Movement Control from Real-World Drone Videos},
  journal   = {arXiv preprint},
  year      = {2024},
}
```

