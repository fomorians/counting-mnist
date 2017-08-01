from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow-gpu==1.1.0',
    'Pillow==4.0.0',
    'scikit-image==0.12.3',
    'termcolor==1.1.0',
    'scipy==0.17.0'
]

setup(name='counting_mnist',
      version='0.0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      include_package_data=True)
