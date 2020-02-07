from setuptools import setup, find_packages
from codecs import open
from os import path
import re

package_name = "matchernet"

root_dir = path.abspath(path.dirname(__file__))


def _requirements():
    return open(path.join(root_dir, 'requirements.txt')).read().splitlines()


with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    license = re.search(r'__license__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author = re.search(r'__author__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    author_email = re.search(r'__author_email__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)
    url = re.search(r'__url__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

assert version
assert license
assert author
assert author_email
assert url

with open('readme.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name=package_name,
    packages=find_packages(exclude=('tests', )),
    version=version,
    license=license,
    install_requires=_requirements(),
    author=author,
    author_email=author_email,
    url=url,
    description='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='kalman filter, neural net',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
