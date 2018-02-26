#!/usr/bin/env python
#import MYUTILS import PWM; import TFHELP;
from setuptools import setup
from setuptools import find_packages
setup(
     name='CisRegModels',    # This is the name of your PyPI-package.
     version='1.0',                          # Update the version number for new releases
     description='Various scripts for processing GPRA data, as well as making models of cis-regulatory logic from GPRA data.',
     packages=find_packages(),
     url='https://github.com/Carldeboer/CisRegModels',
     install_requires=['tensorflow','numpy','datetime'],
     scripts=['collapsePromoters.py','mergeSeqsByBowtie.py','seqsToOHC.py','translateSequencesByDict.py','alignFastqsIntoSeqs.py','makeThermodynamicEnhancosomeModel.py', 'predictThermodynamicEnhancosomeModel.py']                  # The name of your scipt, and also the command you'll be using for calling it
)
