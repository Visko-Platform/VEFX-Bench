from setuptools import setup, find_packages

setup(
    name="vefx-reward",
    version="0.1.0",
    description="VEFX-Reward: A reward model for video editing quality assessment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/taco-group/VEFX-Bench",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "huggingface_hub>=0.20.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "requests",
        "packaging",
        "decord>=0.6.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
