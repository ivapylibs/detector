#!/usr/bin/python
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
 
