# <a href="https://henrywjl.github.io/frequency-guidance-operator/">Frequency Guidance Operator (FGO)</a>

<a href="https://henrywjl.github.io/frequency-guidance-operator/"><strong>Project Website</strong></a>
|
<a href="https://arxiv.org/abs/2605.27919"><strong>arXiv</strong></a>
|
<a href="https://drive.google.com/file/d/1mVgyPEJnNJF142NkGDUb2-FUOafl658y/view?usp=sharing"><strong>Video</strong></a>
|
<a href="https://drive.google.com/drive/folders/17WUGhxG69ddA7HNqqN5lF0zIXbS6Eg3-?usp=drive_link"><strong>Data</strong></a>

**[Junlin Wang](https://henrywjl.github.io/)** **(Solo Author!)**

<p align="center"><img src="media/teaser.gif" alt="drawing" width="100%"/></p>

## 🔧 Installation
#### 1. Create a Conda Environment
```bash
conda create -n fgo_env python=3.8 -y
conda activate fgo_env
```

#### 2. Install PyTorch and TorchVision
```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```
**Note:** If you are using a different CUDA version or hardware setup, please find the appropriate installation command on the [official PyTorch website](https://pytorch.org/get-started/locally/).

#### 3. Install MuJoCo
First, download and extract MuJoCo to your `~/.mujoco` directory:
```bash
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz
```
Next, configure your environment variables so your system can find MuJoCo. You can add these directly to your `~/.bashrc` by running the following commands:
```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```
 
## 💻 Reproducing Simulation Benchmark Results
#### 1. Collect Expert Demonstrations
We use expert policies to collect demonstrations from simulated environments. You may find the following repositories useful for generating your own datasets:
* **[Sim Demo Collector](https://github.com/HenryWJL/sim_demo_collector)**: Our custom package for collecting data in the [Robosuite](https://github.com/ARISE-Initiative/robosuite) and [MimicGen](https://github.com/NVlabs/mimicgen) environments.
* **[3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy)**: Provides tutorials for collecting data in the [Adroit](https://github.com/aravindr93/hand_dapg) and [DexArt](https://github.com/Kami-code/dexart-release) environments.

#### 2. Train Policies
The training code is located in `scripts/train.py`. For example, to train the FGO policy on the Robosuite Lift task:
```bash
python scripts/train.py --config-name=fgo_dp3.yaml \
    task=robosuite_lift \
    task.dataset.zarr_path=<PATH_TO_DATASET> \
    training.device="cuda:0" \
    training.seed=0 \
    training.num_epochs=3000 \
    dataloader.batch_size=512
```
**Note:** This automatically creates a subdirectory under `data/outputs/` where configuration files, logs, and checkpoints are saved. To track your training runs with Weights & Biases, simply append `training.use_wandb=true` to the command.

#### 3. Evaluate Pretrained Policies
Once you have a fully trained policy, you can evaluate its performance in the simulated environments:
```bash
python scripts/eval.py -t robosuite_lift -p fgo_dp3 -c <PATH_TO_CHECKPOINT>
```

## 🦾 Real Robot
### Hardware Requirements
* **[UFACTORY xArm 7 Manipulator](https://www.ufactory.us/xarm)** (If you are using an xArm 5 or xArm 6, you must modify the task configuration files located in `fgo/config/task/`)
* **[UFACTORY xArm Gripper](https://www.ufactory.us/product/ufactory-xarm-gripper)**
* **[StereoLabs ZED 2 Camera](https://www.stereolabs.com/en-za/products/zed-2)** (Requires a USB-C cable and a stable camera stand)
* **[Meta Quest 3](https://www.meta.com/quest/quest-3)** (Used for teleoperation and human demonstration collection.)

### Software Requirements
* **Operating System:** Ubuntu 20.04 or 22.04
* **Python Environment:** Conda is highly recommended
* **xArm SDK:** The official Python wrapper (`pip install xarm-python-sdk`) 
* **[ZED SDK](https://www.stereolabs.com/developers/release):** You must install the system-level C++ SDK first, and then compile the `pyzed` Python wrapper using the provided `get_python_api.py` script

## ❤️ Acknowledgement
* Our DP3 implementation is adapted from [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy).
* Our DiT-Policy implementation is adapted from [Unified World Model](https://github.com/WEIRDLabUW/unified-world-model/tree/main).
* Our FreqPolicy implementation is adapted from [FreqPolicy](https://github.com/4DVLab/Freqpolicy).

## 💬 Citation
```
@article{wang2026fgo,
  author={Wang, Junlin},
  title={Frequency-Guided Action Diffusion via Sub-Frequency Manifold Traversal},
  journal={arXiv preprint},
  year={2026},
```
