from setuptools import setup
from setuptools import find_packages

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name='eigenforms',
    version='0.2.1',
    description='Eigenforms plotter for Nastran SOL103',
    author='slepoi_kamin and brothers E',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    scripts=['EigenForms']
)
