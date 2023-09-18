from setuptools import setup, find_packages

setup(
    name='lm4hpc',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'lm4hpc': ['config.json']},
    install_requires=[
        'torch',
        'transformers',
        'torchvision',
        'accelerate',
        'openai',
        'tiktoken',
        'langchain',
        'faiss-cpu',
        'sentence_transformers',
        'instructorembedding',        
    ],
)
