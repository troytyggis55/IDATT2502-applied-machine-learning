from setuptools import setup, find_packages

setup(
    name="self_play",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["gymnasium==0.29.0", "numpy==2.1.3, pygame==2.1.0"]
)