from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "mujoco",
        "wandb",
        "moviepy",
        "imageio",
        "xxhash",
        "einops",
        "termcolor",
        "setproctitle",
        "torch==2.7.0",
        "tensordict==0.7.2",
        "torchrl==0.7.2",
        "onnxscript",
        "onnxruntime",
        # Isaac Sim 4.5 / Isaac Lab 2.2 require numpy 1.x; ml_dtypes>=0.6 (pulled in via onnx) requires numpy>=2.
        "numpy<2",
        "ml_dtypes<0.6",
        # Must match the warp bundled with Isaac Sim 4.5; a newer pip warp shadows it and breaks Kit extensions.
        "warp-lang==1.5.0",
        # Rotation.from_quat/as_quat(scalar_first=...) requires scipy>=1.14.
        "scipy>=1.14",
        "tqdm",
        "prettytable",
    ],
)
