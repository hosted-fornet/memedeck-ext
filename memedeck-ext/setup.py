#!/usr/bin/env python3

import setuptools

install_requires = [
    'msgpack>=1.0.7',
    'numpy>=1.20.1',
    'tensorflow>=2.11.1',
    'websockets>=12.0',
]

setuptools.setup(
    name='memedeck-ext',
    version='0.1.0',
    author='Holium',
    install_requires=install_requires,
)
