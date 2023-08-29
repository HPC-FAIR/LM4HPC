from setuptools import setup, find_packages

setup(
    name='lm4hpc',
    version='0.1',
    packages=find_packages(where='./lm4hpc'),
    package_dir={'': '.',
                 "lm4hpc": "./lm4hpc"},
    install_requires=[
        'torch',
        'transformers',
    ],
)
