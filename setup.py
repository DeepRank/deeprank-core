#!/usr/bin/env python

import os

from setuptools import (find_packages, setup)

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit DeepRank-GNN/__version__.py
version = {}
with open(os.path.join(here, 'deeprank_gnn', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name='DeepRank-GNN-2',
    version=version['__version__'],
    description='Graph Neural network Scoring of protein-protein conformations',
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    author=["Giulia Crocioni", "Coos Baakman", "Daniel Rademaker", "Gayatri Ramakrishnan", "Sven van der Burg", "Li Xue", "Daniil Lepikhov"],
    author_email='g.crocioni@esciencecenter.nl, Coos.Baakman@radboudumc.nl',
    url='https://github.com/DeepRank/deeprank-gnn-2',
    packages=find_packages(),
    package_dir={'deeprank_gnn': 'deeprank_gnn'},
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='deeprank_gnn_2',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English', 'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    test_suite='tests',

    setup_requires=['torch>=1.11.0'],

    # not sure if the install of torch-geometric will work ..
    install_requires=['torch-cluster>=1.6.0', 'torch-sparse>=0.6.13',
        'torch-geometric>=2.0.4', 'torch-scatter>=2.0.9', 'torch-spline-conv>=1.2.1',
        'numpy >= 1.21.5', 'scipy >= 1.7.3', 'h5py >= 3.6.0',
        'networkx == 2.6.3', 'matplotlib >= 3.5.1', 'pdb2sql >= 0.5.1', 'sklearn',
        'chart-studio >= 1.1.0', 'BioPython >= 1.79', 'python-louvain >= 0.16',
        'markov-clustering >= 0.0.6.dev0',
        'tqdm >= 4.63.0', 'freesasa >= 2.1.0',
        'tensorboard >= 2.9.0', 'torchvision >= 0.12.0'
    ],
    extras_require={
        'dev': ['prospector[with_pyroma]', 'yapf', 'isort'],
        'doc': ['recommonmark', 'sphinx', 'sphinx_rtd_theme'],
        'test': ['coverage', 'pycodestyle', 'pytest',
                 'pytest-cov', 'pytest-runner', 'coveralls'],
    })
