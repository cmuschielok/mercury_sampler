from setuptools import setup

from mercury_sampler import __version__, __author__, __email__


setup(
    name='mercury_sampler',
    version=__version__,
    packages=['mercury_sampler'],
    url='',
    license='',
    author=__author__,
    author_email=__email__,
    description='Sampling of pore networks with atoms and linear molecules. '
                'Mercury, because it is the host-guest sampler, hg-sampler...',
    entry_points={'console_scripts': ['hg-sampler = mercury_sampler.cli:sample']}
)
