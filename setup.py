#! /usr/bin/env python
# -*- coding: utf-8 -*_>
from distutils.core import setup
import setuptools

setup(
    name='apireplacement',  # 包的名字
    version='0.0.1',  # 版本号
    description='a tool api_replacement that can help developers discover low-efficiency APIs in the code and recommend higher-efficiency APIs',  # 描述
    author='tangshan',  # 作者
    author_email='709166298@qq.com',  # 你的邮箱**
    url='https://github.com/civilTANG/APIreplace',  # 可以写github上的地址，或者其他地址
    packages=setuptools.find_packages(exclude=['apireplacement']),  # 包内需要引用的文件夹

    # 依赖包
    install_requires=[
        'astor',
        're',
        ' ast'
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
