#!/usr/bin/env python
from setuptools import setup
setup(
     name='CisRegModels',    # This is the name of your PyPI-package.
     version='1.0',                          # Update the version number for new releases
     scripts=['collapsePromoters.py','mergeSeqsByBowtie.py','seqsToOHC.py','translateSequencesByDict.py','alignFastqsIntoSeqs.py','makeThermodynamicEnhancosomeModel.py', 'predictThermodynamicEnhancosomeModel.py']                  # The name of your scipt, and also the command you'll be using for calling it
)
