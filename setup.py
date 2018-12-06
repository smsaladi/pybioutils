from setuptools import setup

setup(
  name='bioutils-saladi',
  packages=['bioutils'],
  author='Shyam Saladi',
  author_email='saladi@caltech.edu',
  url='https://github.com/smsaladi/pybioutils',
  install_requires=['numpy', 'pandas', 'biopython'],
  setup_requires=['pytest-runner'],
  tests_require=['pytest', 'pandas'],
)

