# setup.py
from setuptools import setup, find_packages

VERSION = '0.0.1a1'
DESCRIPTION = 'A Deep Learning Package for Entry Level Tinkerers'

setup(
    name='neo-dl',
    version= VERSION,
    default_version= VERSION,
    packages=find_packages(),
    author='vxnuaj',
    author_email='jv.100420@gmail.com',
    description= DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vxnuaj/neo-dl',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
    ],
    install_requires=['numpy', 'pandas', 'matplotlib'],
    tests_require=['unittest'],
)