# setup.py
from setuptools import setup, find_packages

setup(
    name="dibs_torch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
    ],
)