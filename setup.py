import os

from setuptools import find_packages, setup


# Read the README file
def read_file(fname):
    with open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="lobpy",
    version="1.2.0",
    author="Mattia",
    description="Limit Order Book in Python",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/mattermat/lob.py",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    keywords="limit order book lob trading finance",
    project_urls={
        "Bug Reports": "https://github.com/mattermat/lob.py/issues",
        "Source": "https://github.com/mattermat/lob.py",
    },
)
