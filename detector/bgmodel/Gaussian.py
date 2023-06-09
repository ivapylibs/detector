#================================ Gaussian ===============================
"""
  @class Gaussian

  @brief    Implements a single Gaussian background model.

  No doubt this implementation exists in some form within the OpenCV or
  BGS libraries, but getting a clean, simple interface from these libraries
  is actually not as easy as implementing from existing Matlab code.
  Plus, it permits some customization that the library implementations
  may not have.

  Inputs:
    mu          - the means of the Gaussian models.
    sigma       - the variance of the Gaussian models.
    weights     - the weights of the Gaussian models.
    parms       - [optional] configuration instance with parameters specified.

  Fields of the parms structure:
    sigma       - Initial variance to use if sigma is empty.
    thresh      - Threshold for determining foreground.
    alpha       - Update rate for mean and variance.

    A note on the improcessor.  If the basic version is used, then it
    performs pre-processing.  If a triple version is used, then the
    mid-processor will perform operations on the detected part rather
    than the default operations.  The mid-processor can be used to test
    out different options for cleaning up the binary data.
"""
#================================ Gaussian ===============================
#
# @file     Gaussian.py
#
# @author   Jun Yang,
# @author   Patricio A. Vela,   pvela@gatech.edu
#
# @date     2009/07/XX      [original Matlab]
# @date     2012/07/07      [taken Matlab version]
# @date     2023/06/08      [converted to python]
#
# Version   1.0
#
# Notes:    set tabstop = 4, indent = 2, 80 columns.
#
#================================ Gaussian ===============================

from yacs.config import CfgNode
import numpy as np

from detector.inImage import inImage



class CfgSGM(CfgNode):
  '''!
  @brief  Configuration setting specifier for Gaussian BG model.

  @note   Currently not using a CfgBGModel super class. Probably best to do so.
  '''
  #============================= __init__ ============================
  #
  '''!
  @brief        Constructor of configuration instance.

  @param[in]    cfg_files   List of config files to load to merge settings.
  '''
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    if (init_dict == None):
      init_dict = CfgD435.get_default_settings()

    super().__init__(init_dict, key_list, new_allowed)

    # self.merge_from_lists(XX)

  #========================= get_default_settings ========================
  #
  # @brief    Recover the default settings in a dictionary.
  #
  @staticmethod
  def get_default_settings():
    '''!
    @brief  Defines most basic, default settings for RealSense D435.

    @param[out] default_dict  Dictionary populated with minimal set of
                              default settings.
    '''

  default_dict = dict(tauSigma = 3.5, minSigma = 50, alpha = 0.05, 
    init = dict( sigma = 20 )  )
  return default_dict




class Gaussian(inImage):

  #============================== Gaussian =============================
  #
  #
  def __init__(self, processor = None, bgMod = None, bgCfg):
    '''!
    @brief  Constructor for single Gaussian model background detector.
    
    @param[in]  bgMod   The model or parameters for the detector.
    @param[in]  bgCfg   The model or parameters for the detector.
    '''
    super(inCorner, self).__init__(processor)

    # @todo Fix. What does this mean?
    self.bgModel = bgMod

    # @todo Add in bgCFG
    if  bgCfG not None:
      self.config = bgCfG
    else
      self.config = CfGSGM.get_default_settings()

    # @todo Fix me.

  def set(self):
    pass

  def get(self):
    pass

  def predict(self):
    pass

  def measure(self, I):
    '''!
    @brief    Takes image and generates the detection result.
  
    @param[in]    I   Image to process.
    '''
    if self.improcessor is not None: 
      I = self.improcessor.pre(I)
  
    pI = reshape( I, [prod(imsize(1:2)), imsize(3)] );
    sqeI = ((mu - pI).^2)./sigma;
    errI = max(sqeI, [], 2);
  
    bgI  = reshape( errI < bgp.SigmaT, imsize(1:2) );
  
    if self.improcessor is not None:
      bgI = bgp.improcessor.post(bgI)
  

  def correct(self):
    pass


  def adapt(self):

    # MATLAB CODE:

    fginds = find(~bgI);                          % Get foreground pixels.
  
    oldmu    = mu(fginds,:);                      % Save their old values.
    oldsigma = sigma(fginds,:);
  
    mu    = (1-bgp.alpha)*mu + bgp.alpha*pI;      % Update mean and sigma.
    sigma = (1-bgp.alpha)*sigma + bgp.alpha*(pI-mu).^2;
    for ii=1:size(sigma,2)                        % Impose min sigma constraint.
      sigma(:,ii) = max(sigma(:,ii), bgp.minSigma(ii));
  
    mu(fginds,:) = oldmu;                         % Revert foreground estimates
    sigma(fginds,:) = oldsigma;                   % to their old values.


  def process(self, I):
  
    self.predict();
    self.measure(I);
    self.correct();
    self.adapt();

emptystate
emptydebug
getstate
getdebug
displayState
displayDebug
info
save    # Save given file.
saveTo  # Save given HDF5 pointer. Puts in root.
saveCfG # Save to YAML file.

STATIC FUNCTIONS
load
loadFrom



#
#================================ Gaussian ===============================
