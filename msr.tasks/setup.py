from setuptools import setup, find_packages

setup(
    name="msr-tasks",  
    version="0.1.0",
    author="wl",
    description="RL tasks for the MSR project.",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.10",
)