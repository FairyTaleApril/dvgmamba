# DVGMamba

## Installation
1. **Create and activate a Conda environment**:
    ```sh
    conda create -n dvg python=3.10 -y
    conda activate dvg
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    conda install -c conda-forge ffmpeg -y
    conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    conda install -c conda-forge ffmpeg -y
    wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    pip install -r requirements.txt
    ```

2. **Download evaluation data**
   
    For real city 3D scans from Google Earth, please download from this [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EhrvMtW9ow5KrpfPJlAnJ9wBjaaYqNEKx98NOXGFteJ3pg?e=d99AG4).

    For synthetic natural scenes, please download from this [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EgQWiB64W6dCsuOko_UoNQoB9Zj4cb-SSlqLFdVZITJT7Q?e=MBvCGx).


3. **Download training data**
   
    You can download DroneMotion-99k dataset from this [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/EcHhl1KtZrdHn4wkDJ9Kcg4BtwQCP3f3hKUHS7PArhprnw?e=SRkFjl). 

    Please note that this dataset is different from mine.
