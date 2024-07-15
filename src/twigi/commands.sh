#!/bin/bash

# we don't want to run this file, so start with an exit command
exit

# build new version
python setup.py bdist_wheel

# upload to pypi with my api key (will need to be input manually)
twine upload dist/* --verbose

# install locally
python setup.py install
