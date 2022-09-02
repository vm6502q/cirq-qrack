#Adapted from https://github.com/Qiskit/qiskit-aer/blob/master/setup.py

import os
from distutils.core import setup

requirements = [
    'pyqrack>=0.8.0'
]

# Handle version.
VERSION = "0.2.0"

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(
    name='cirqqrack',
    version=VERSION,
    packages=['qrack'],
    description="Cirq Qrack Plugin - Qrack High-Performance GPU simulation for Cirq",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/vm6502q/cirq-qrack",
    author="Daniel Strano",
    author_email="stranoj@gmail.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    keywords="cirq qrack simulator quantum addon backend",
    install_requires=requirements
)
