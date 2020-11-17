import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


requirements = [
    'torch>=1.3.1',
    'torch-dct',
    'numpy>=1.15.4',
    'scipy>=1.1.0',
    'absl-py>=0.1.9',
    'mpmath>=1.1.0',
]

requirements_dev = [
    'Pillow',
    'nose'
]


setup(
    name="deep_evidential_regression_loss_pytorch",
    version="0.0.1",
    url="https://github.com/deebuls/deep_evidential_regression_loss_pytorch",
    license='Apache 2.0',
    author="Deebul S. Nair",
    author_email="deebul.nair@h-brs.de",
    description="A Loss function which predicts posterior distribution for regression problems",
    long_description=read("README.md"),
    #package_dir={'':'deep_evidential_regression_loss_pytorch'},  # Optional
    #packages=find_packages(where=('deep_evidential_regression_loss_pytorch'),exclude=('tests',)),
    packages=find_packages(exclude=('tests',)),
    install_requires=requirements,
    extras_require={
        'dev': requirements_dev
    },
)

