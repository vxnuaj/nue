# setup.py
from setuptools import setup, find_packages

VERSION = '0.0.4'
DEFAULTV = '0.0.4'
DESCRIPTION = 'A deep learning framework sculpted for seamless implementation of models, geared towards entry level learners. '

setup(
    name='nue',
    version= VERSION,
    default_version= DEFAULTV,
    packages=find_packages(exclude = ['examples', 'docs', 'dist', 'test']),
    exclude_package_data={'': ['.gitignore', '.gitattributes']},
    author='vxnuaj',
    author_email='jv.100420@gmail.com',
    description= DESCRIPTION,
    long_description=open('PYPI.md').read(),
    long_description_content_type='text/markdown',
    url='https://vxnuaj.github.io/nue/',
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