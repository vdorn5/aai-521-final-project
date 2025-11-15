from setuptools import setup, find_packages

setup(
    name="spine-score",  # Name of your project/package
    version="0.1.0",
    description="Spine MRI computer vision ETL and modeling project",
    author="Victoria Dorn",
    packages=find_packages(where="spine-score"),  # Tells setuptools to look inside src/
    package_dir={"": "spine-score"}  # Maps root package to src/
)
