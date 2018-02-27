# CisRegModels
Scripts for building computational models of gene regulation with tensorflow

#Installation
We recommend using Anaconda and pip to install this.  Due to some dependencies with other (poorly written/distributed) packages, installation is not one step, but it is nearly so

This module requires tensorflow (ideally with GPU support since execution time is orders of magnitude slower without GPUs), as well as several other standard python modules

1. Install CisRegModels:
pip install git+https://github.com/Carldeboer/CisRegModels

2. Include other dependencies
2.1 Download other dependencies:
`wget https://raw.githubusercontent.com/Carldeboer/BigWig-Tools/master/MYUTILS.py `
2A Find install directory:


