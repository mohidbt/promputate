"""
Setup script for Promputate library
"""

from setuptools import setup, find_packages

# Read version from package
with open("promputate/__init__.py", "r") as f:
    content = f.read()
    for line in content.split("\n"):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="promputate",
    version=version,
    author="Promputate Team",
    author_email="promputate@example.com",
    description="Genetic Algorithm for Prompt Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/promputate/promputate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=[
        "deap>=1.3.1",
        "nltk>=3.8.1",
        "spacy>=3.7.0",
    ],
    extras_require={
        "llm": ["openai>=1.35.0", "anthropic>=0.25.0"],
        "analysis": ["pandas>=2.0.0", "statsmodels>=0.14.0"],
        "api": ["fastapi>=0.100.0", "uvicorn>=0.20.0"],
        "ui": ["streamlit>=1.25.0"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.20.0"],
        "all": [
            "openai>=1.35.0", "anthropic>=0.25.0",
            "pandas>=2.0.0", "statsmodels>=0.14.0",
            "fastapi>=0.100.0", "uvicorn>=0.20.0",
            "streamlit>=1.25.0",
            "pytest>=7.0.0", "pytest-asyncio>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "promputate=promputate.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "promputate": ["config/*.yaml", "data/*.json"],
    },
) 