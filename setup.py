import os
from setuptools import setup, find_packages

__version__ = "0.1.0"

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

setup(
    name="fgo",
    version=__version__,
    description="Frequency Guidance Operator (FGO).",
    author="Junlin Wang",
    author_email="wangjl@seas.upenn.edu",
    url="https://henrywjl.github.io/frequency-guidance-operator/",
    packages=find_packages(include=["fgo*"]),
    python_requires=">=3.10",
    setup_requires=["setuptools>=62.3.0"],
    include_package_data=True,
    install_requires=[
        "Cython==0.29.35",
        "dill==0.3.5.1",
        "diffusers==0.35.2",
        "dm_control==1.0.37",
        "einops==0.8.0",
        "fpsample==1.0.2",
        "gpustat==1.1.1",
        "gymnasium==1.2.3",
        "hydra-core==1.2.0",
        "imageio==2.37.2",
        "ipdb==0.13.13",
        "moviepy==1.0.3",
        "matplotlib==3.10.8",
        "numba==0.58.1",
        "numpy==1.26.4",
        "numcodecs==0.11.0",
        "open3d==0.19.0",
        "omegaconf==2.3.0",
        "patchelf==0.17.2.0",
        "Pillow==12.0.0",
        "scipy==1.15.0",
        "tqdm==4.67.3",
        "termcolor==3.3.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torch-dct==0.1.6",
        "wandb==0.25.0",
        "zarr==2.17.0",
        
        # Third-party local dependencies
        f"mujoco-py @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/mujoco-py-2.1.2.14')}"
        f"dexart @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/dexart-release')}",
        f"gym @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/gym-0.21.0')}",
        f"mj_envs @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/rrl-dependencies/mj_envs')}",
        f"mjrl @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/rrl-dependencies/mjrl')}",
        f"mimicgen @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/mimicgen')}",
        f"pytorch3d @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/pytorch3d_simplified')}",
        f"robosuite @ file://localhost/{os.path.join(ROOT_PATH, 'third_party/robosuite')}",
    ],
)