from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rl-playground",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Professional reinforcement learning framework for CartPole and other environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rl_playground",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "isort",
            "pre-commit",
        ],
        "vision": [
            "timm",
            "albumentations",
        ],
    },
    entry_points={
        "console_scripts": [
            "rl-train=train:main",
            "rl-eval=evaluate:main",
            "rl-demo=demo:main",
        ],
    },
)