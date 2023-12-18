#================================== base =================================
"""!
@brief      Base class for the state_parser and the the activity parser

"""
#================================== base =================================
"""
@author     Yiye Chen.          yychen2019@gatech.edu
@date       2021/08/19          [created]
@date       2023/12/15          [moved to detector package]
"""
#================================== base =================================


# @note There is already a detector base class.  Shouldn't this be that?
#       Better put, is this class even needed?

class Base(object):

    def __init__(self):
        self.x = None
        pass

    def process(self, signal):
        """
        Process the new income signal
        """
        self.predict()
        self.measure(signal)
        self.correct()
        self.adapt()

    def predict(self):
        return None

    def measure(self, signal):
        return None

    def correct(self):
        return None
    
    def adapt(self):
        return None



#
#================================== base =================================
