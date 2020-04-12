#! /usr/bin/env python
# -*- coding: utf-8 -*_>
from distutils.core import setup
import setuptools

setup(
    name='apireplacement',  
    version='0.0.1', 
    description='a tool api_replacement that can help developers discover low-efficiency APIs in the code and recommend higher-efficiency APIs',  # 描述
    author='tangshan',  
    author_email='709166298@qq.com',  
    url='https://github.com/civilTANG/APIreplace', 
    packages=setuptools.find_packages(exclude=['apireplacement']), 


    install_requires=[
        'astor',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: Microsoft'  # 你的操作系统
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',  # BSD认证
        'Programming Language :: Python',  # 支持的语言
        'Programming Language :: Python :: 3',  # python版本 。。。
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
    zip_safe=True,
)
