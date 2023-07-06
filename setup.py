from setuptools import setup, find_packages

setup(
    name="ddpo-pytorch",
    version="0.0.1",
    packages=["ddpo_pytorch"],
    python_requires=">=3.10",
    install_requires=[
        "ml-collections",
        "absl-py",
        "diffusers[torch]==0.17.1",
        "wandb",
        "torchvision",
        "inflect==6.0.4",
        "pydantic==1.10.9",
        "transformers==4.30.2",
    ],
)
