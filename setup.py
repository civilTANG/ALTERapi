#! /usr/bin/env python
# -*- coding: utf-8 -*_>
from distutils.core import setup
import setuptools

setup(
    name='alterapi',  
    version='1.0',
    description='A tool that can help developers discover low-efficiency API usages in the code and recommend high-efficiency alternatives.',
    author='tangshan',  
    author_email='709166298@qq.com',  
    url='https://github.com/civilTANG/APIreplace', 
    packages=setuptools.find_packages(exclude=[]), 


    install_requires=[
        'astor',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: Microsoft'  
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python', 
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.4', 
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    zip_safe=True,
)
