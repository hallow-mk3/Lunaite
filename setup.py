from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lunaite",
    version="3.0.0",
    author="Swasthik Shetty",
    author_email="swasthik.mk3@gmail.com",
    description="Lunaite — Universal Modular AI Architecture Framework (Sparse MoE, Cognitive Deliberation, Persistent Memory, Autonomous Agent Tools)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hallow-mk3/Lunaite",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "pydantic>=2.6.0",
        "psutil>=5.9.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "desktop": ["pyautogui>=0.9.54", "pyperclip>=1.8.2"],
        "voice": ["pyttsx3>=2.90", "SpeechRecognition>=3.10.0"],
        "all": ["pyautogui>=0.9.54", "pyperclip>=1.8.2", "pyttsx3>=2.90", "SpeechRecognition>=3.10.0", "bitsandbytes>=0.43.0"]
    },
    entry_points={
        "console_scripts": [
            "lunaite=lunaite.cli:main",
        ],
    },
    include_package_data=True,
)
