import os
from setuptools import setup, find_packages

setup(
    name="portfolio_layer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "cvxpy",
        "pytest"
    ],
)
