from setuptools import setup, find_packages

setup(
    name="stemOrchestrator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",  # Or any other dependencies
    ],
    entry_points={
        "console_scripts": [
            "run-acquisition=scripts.run_acquisition:main",
        ],
    },
)

