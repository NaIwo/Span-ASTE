from setuptools import setup, find_packages

setup(
    name='aste',
    version='1.0',
    description='Astpect Sentiment Triplet Extraction',
    author='Iwo Naglik',
    author_email='',
    packages=find_packages(),
    include_package_data=True,
    package_data={'aste': ['*.yml']},
    install_requires=[
        "numpy>=1.22.3",
        "pandas>=1.4.2",
        "torch>=1.11.0",
        "torchmetrics>=0.7.3",
        "tqdm>=4.64.0",
        "transformers>=4.23.1"
        ]
)
