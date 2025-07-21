from setuptools import setup, find_packages

setup(
    name='view-of-delft-dataset',
    version='1.0.2',
    description='View of Delft dataset Python package',
    author='TUDelft IV',
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        'numpy==1.19',
        'numba',
    ],
    python_requires='==3.7.*',
)

