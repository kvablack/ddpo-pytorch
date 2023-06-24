from setuptools import setup, find_packages

setup(
    name='ddpo-pytorch',
    version='0.0.1',
    packages=["ddpo_pytorch"],
    install_requires=[
        "ml-collections", "absl-py"
    ],
)