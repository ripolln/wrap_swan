from distutils.core import setup
import os

import wswan

def _strip_comments(l):
    return l.split('#', 1)[0].strip()

def _pip_requirement(req):
    if req.startswith('-r '):
        _, path = req.split()
        return reqs(*path.split('/'))
    return [req]

def _reqs(*f):
    return [
        _pip_requirement(r) for r in (
            _strip_comments(l) for l in open(
                os.path.join(os.getcwd(), *f)).readlines()
        ) if r]

def reqs(*f):
    """Parse requirement file.
    Returns:
        List[str]: list of requirements specified in the file.
    Example:
        reqs('default.txt')          # ./default.txt
        reqs('extras', 'redis.txt')  # ./extras/redis.txt
    """
    return [req for subreq in _reqs(*f) for req in subreq]

def install_requires():
    """Get list of requirements required for installation."""
    return reqs('requirements.txt')

setup(
    name             = 'wswan',
    version          = wswan.__version__,
    description      = wswan.__description__,
    long_description = open('README.md').read(),
    keywords         = wswan.__keywords__,
    author           = wswan.__author__,
    author_email     = wswan.__contact__,
    url              = wswan.__url__,
    license          = 'LICENSE.txt',
    python_requires  = ">=3.7.10",
    install_requires = install_requires(),
    include_package_data = True,
    packages         = ['wswan'],
    package_data     = {'wswan' : ['wswan/resources/*',
                                   'wswan/resources/swan_bin/*']},
    scripts          = [],
)

