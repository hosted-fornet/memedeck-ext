#!/usr/bin/env python3

import setuptools

install_requires = [
    'msgpack>=1.0.7',
    'numpy>=1.20.1',
    'tensorflow>=2.11.1',
    'torch>=2.0.1',
    'websockets>=12.0',
]

setuptools.setup(
    name='ml-ext',
    version='0.1.0',
    author='Kinode DAO',
    url='https://kinode.org',
    install_requires=install_requires,
)
