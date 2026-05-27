# <a href="https://henrywjl.github.io/frequency-guidance-operator/">Frequency Guidance Operator (FGO)</a>

<a href="https://henrywjl.github.io/frequency-guidance-operator/"><strong>Project Website</strong></a>
|
<a href=""><strong>arXiv</strong></a>
|
<a href="https://drive.google.com/file/d/1mVgyPEJnNJF142NkGDUb2-FUOafl658y/view?usp=sharing"><strong>Video</strong></a>
|
<a href="https://drive.google.com/drive/folders/17WUGhxG69ddA7HNqqN5lF0zIXbS6Eg3-?usp=drive_link"><strong>Data</strong></a>

<a href="https://henrywjl.github.io/">Junlin Wang</a>

<p align="center"><img src="media/teaser.gif" alt="drawing" width="80%"/></p>

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
**Note**: If you are using a different CUDA version or hardware setup, please find the appropriate installation command on the [official PyTorch website](https://pytorch.org/get-started/locally/).

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
 
## 💻 Training
### Downloading a Dataset
We provide a new dataset spanning 8 manipulation tasks across 3 different robots from the RLBench and Robosuite benchmarks. [Here](https://github.com/HenryWJL/icon/tree/main/icon/configs/task) is a complete list of tasks. To download our full dataset:
```bash
python scripts/download_dataset.py
```
If you'd like to download the data for a specific task (e.g., *Close Drawer*):
```bash
python scripts/download_dataset.py -t close_drawer
```
You can also download the dataset directly from the [Hugging Face](https://huggingface.co/datasets/HenryWJL/icon).

### Running on a Device 
Now it’s time to give it a try! You can run `scripts/train.py` to train any algorithm on any task you like.
For example, to train a CNN-based diffusion policy coupled with ICon on the *Open Box* task:
```bash
python scripts/train.py task=open_box algo=icon_diffusion_unet
```
This will automatically create a subdirectory at `outputs/TASK_NAME/ALGO_NAME/YYYY-MM-DD/HH-MM-SS`, where configuration files, log files, and checkpoints will be saved. If you want to run on a different device with a different seed, simply append the desired arguments to the command:
```bash
python scripts/train.py task=open_box algo=icon_diffusion_unet train.device=cuda:0 train.seed=100
```
To enable Weights & Biases:
```bash
wandb login
python scripts/train.py task=open_box algo=icon_diffusion_unet train.device=cuda:0 train.seed=100 train.wandb.enable=true
```

## ⏳ Evaluation
Once you have obtained a well-trained policy, you can evaluate its performance in the simulated environments. For example, to evaluate a Transformer-based diffusion policy agumented with ICon on the *Close Microwave* task for 50 episodes: 
```bash
python scripts/eval_sim_robot.py -t close_microwave -a icon_diffusion_transformer -c PATH_TO_YOUR_CHECKPOINT -ne 50
```
Recorded episode videos will be saved in `videos/TASK_NAME/ALGO_NAME`. For real-time visualization, set the rendering mode to "human":
```bash
python scripts/eval_sim_robot.py -t close_microwave -a icon_diffusion_transformer -c PATH_TO_YOUR_CHECKPOINT -ne 50 -rm human
```
