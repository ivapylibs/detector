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

    #================================ Base ===============================
    #
    def __init__(self):
        """!
        @brief  Instantiate a detector Base activity class object.
        """

        self.x = None
        pass

    #============================== process ==============================
    #
    def process(self, signal):
        """!
        @brief Process the new incoming signal. Establish activity state.
        """

        self.predict()
        self.measure(signal)
        self.correct()
        self.adapt()

    #============================== predict ==============================
    #
    def predict(self):
        """!
        @brief  If transition model known, generate state prediction. Else
                it is most likely a static transition model, which does 
                nothing / keeps prior state.
        """
        return None

    #============================== measure ==============================
    #
    def measure(self, signal):
        """!
        @brief  Generate measurement of activity state from passed signal.

        @param[in]  signal  Current signal of interest for activity detection.
        """
        return None

    #============================== correct ==============================
    #
    def correct(self):
        """!
        @brief  Reconcile prediction and measurement as fitting.
        """
        return None
    
    #=============================== adapt ===============================
    #
    def adapt(self):
        """!
        @brief  Adapt any internal parameters based on activity state, signal,
                and any other historical information.
        """
        return None


#
#================================== base =================================
