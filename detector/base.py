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

from dataclasses import dataclass
import h5py


@dataclass
class detectorState:
    """!
    @brief      Basic version of a detector state.

    The most basic version has space for the detected instance.
    It should be empty/none if no detection.
    """
    x: any = None


class Base(object):
    """!
    @ingroup    Detector
    @brief      Base or root implementation of detector class.
    """

    #================================ Base ===============================
    #
    def __init__(self):
        """!
        @brief  Instantiate a detector Base activity class object.
        """

        self.x = None
        pass

    #---------------------------------------------------------------------
    #-------------------- Detector Processing Routines -------------------
    #---------------------------------------------------------------------

    #============================== predict ==============================
    #
    def predict(self):
        """!
        @brief  Predict next state from current state.

        If transition model known, generate state prediction. Else it is most
        likely a static transition model, which does nothing / keeps prior
        state. The base method employs a static state assumption.  
        """
        pass

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

        Correct state based on measurement and prediction states.  Base method
        does not have correction.
        """
        pass
    
    #=============================== adapt ===============================
    #
    def adapt(self):
        """!
        @brief  Adapt any internal parameters based on activity state, signal,
                and any other historical information.

        Base method does not have adaptation.  Should be customized to
        the implementation of the class.
        """
        return None

    #============================== process ==============================
    #
    def process(self, signal):
        """!
        @brief Process the new incoming signal on full detection pipeline.

        Running the full detection pipeline includes adaptation.
        """

        self.predict()
        self.measure(signal)
        self.correct()
        self.adapt()

    #=============================== detect ==============================
    #
    def detect(self, signal):
        """!
        @brief  Run detection only processing pipeline (no adaptation).

        Good to have if there is a calibration scheme that uses adaptation, then
        freezes the parameters during deployment.  Running detect rather than
        process prevents further model updating while still running the
        rest of the process in a single call.
        """
        self.predict()
        self.measure(signal)
        self.correct()

    #---------------------------------------------------------------------
    #------------------- Detector State Access Routines ------------------
    #---------------------------------------------------------------------

    #============================= emptyState ============================
    #
    def emptyState(self):
        """!
        @brief  Return empty state. Useful if contents needed beforehand.
  
        """
        state = detectorState
        return state

    #============================== getState =============================
    #
    def getState(self):
        """!
        @brief  Return current/latest state. 
  
        """
        state = detectorState(x=self.Ip)
        return state

    #============================= emptyDebug ============================
    #
    def emptyDebug(self):
        """!
        @brief  Return empty debug state information. Useful if contents needed
                beforehand.
    
        """
    
        return None         # For now. just getting skeleton code going.
    
    #============================== getDebug =============================
    #
    def getDebug(self):
        """!
        @brief  Return current/latest debug state information. 
    
        Usually the debug state consists of internally computed information
        that is useful for debugging purposes and can help to isolate problems
        within the implemented class or with downstream processing that may
        rely on assumptions built into this implemented class.
        """
  
        return None         # For now. just getting skeleton code going.
  
    #---------------------------------------------------------------------
    #------------------------- Save/Load Routines ------------------------
    #---------------------------------------------------------------------

    #================================ save ===============================
    #
    def save(self, fileName):    # Save given file.
        """!
        @brief  Outer method for saving to a file given as a string.

        Opens file, preps for saving, invokes save routine, then closes.
        Usually not overloaded.  Overload the saveTo member function.
        """
        fptr = h5py.File(fileName,"w")
        self.saveTo(fptr);
        fptr.close()

    #================================ load ===============================
    #
    @staticmethod
    def load(fileName, relpath = None):    # Load given file.
        """!
        @brief  Outer method for loading file given as a string (with path).

        Opens file, preps for loading, invokes loadFrom routine, then closes.
        Overload to invoke sub-class loadFrom member function.

        @param[in]  fileName    The full or relative path filename.
        @param[in]  relpath     The hdf5 (relative) path name to use for loading.
                                Usually class has default, this is to override.
        """
        fptr = h5py.File(fileName,"r")
        if relpath is not None:
          theInstance = Base.loadFrom(fptr, relpath);
        else:
          theInstance = Base.loadFrom(fptr)
        fptr.close()
        return theInstance

    #=============================== saveTo ==============================
    #
    def saveTo(self, fPtr):    
        """!
        @brief  Empty method for saving internal information to HDF5 file.

        Save data to given HDF5 pointer. Puts in root.
        """
        # Not sure what goes here.  Leaving empty.
        # Maybe eventually save the info strings / structure / dict.
        pass

    #============================== loadFrom =============================
    #
    @staticmethod
    def loadFrom(fPtr):    
        """!
        @brief  Empty method for loading internal information from HDF5 file.

        Load data from given HDF5 pointer. Assumes in root from current file
        pointer location.
        """
        return Base()



#
#================================== base =================================
