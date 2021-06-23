#!/usr/bin/env python
"""
setup
"""
from setuptools import setup, find_packages

version = '0.0.1'

setup(
    name="mindtext",
    version=version,
    author="MindText Core Team",
    url="https://gitee.com/mindspore/mindtext/tree/master/",
    packages=find_packages(exclude="example"),
)
