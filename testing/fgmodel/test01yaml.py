#!/usr/bin/python
#=============================== test01yaml ==============================
'''!
@brief  Short code to confirm and demonstrate use of YAML configuration file.


Execution:
----------
Just run. It saves to a yaml file.
When deon, review the YAML file.
'''
#=============================== test01yaml ==============================

#
# @file     test01yaml.py
#
# @author   Patricio A. Vela        pvela@gatech.edu
# @date     2023/06/XX
#
#
# NOTE:     indent is 4 spaces with conversion.  text is 80 columns.
#
#=============================== test01yaml ==============================


import numpy as np
import detector.fgmodel.Gaussian as SGM 


config = SGM.CfgSGT.builtForRedGlove()

config.init.mu    = [130.0,10.0,50.0]
config.init.sigma = [650.0,150.0,250.0]

print(config.dump())

with open('./test01yaml.yaml','w') as file:
    file.write(config.dump())
    file.close()
 
#
#=============================== test01yaml ==============================
