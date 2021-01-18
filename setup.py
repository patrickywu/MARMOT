import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MARMOT", # Replace with your own username
    version="0.0.1",
    author="Patrick Y. Wu and Walter R. Mebane, Jr.",
    author_email="pywu@umich.edu",
    description="Multimodal representations for images and text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/patrickywu/MARMOT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
