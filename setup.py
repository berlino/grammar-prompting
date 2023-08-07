import sys

from setuptools import setup, find_packages

setup(
    name="neural_lark",
    version="0.1",
    author='bailin',
    description="neural parser",
    packages=find_packages(
        exclude=["*_test.py", "test_*.py", "tests"]
    ),
    install_requires=[
        "tqdm~=4.64.1",
        "wandb>=0.9.4",
        "openai>=0.26.5",
        "numpy>=1.23.5",
        "lark~=1.1.5",
        "torch~=2.0.0",
        "stanza~=1.4.2",
        "google.generativeai~=0.1.0rc3",
        "tiktoken~=0.4.0",
        "nltk~=3.8.1",
        "exrex~=0.11.0",
        "transformers~=4.26.1",
        "overrides~=7.3.1",
        "lark~=1.1.7"
    ],
)