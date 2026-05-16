from setuptools import setup, find_packages

__version__ = "0.1.0"

setup(
    name="fgo",
    version=__version__,
    description="Frequency Guidance Operator (FGO).",
    author="Junlin Wang",
    author_email="wangjl@seas.upenn.edu",
    url="https://henrywjl.github.io/frequency-guidance-operator/",
    packages=find_packages(include=["fgo*"]),
)