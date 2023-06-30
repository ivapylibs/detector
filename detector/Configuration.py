#=============================== Configuration ===============================
'''!

'''
#=============================== Configuration ===============================
#
# @file     Configuration.py
#
#
#=============================== Configuration ===============================


import copy
import io
import logging
import sys
import os
from ast import literal_eval

import yaml

from yacs.config import CfgNode

#
#-----------------------------------------------------------------------------
#========================== Setup Local Environment ==========================
#-----------------------------------------------------------------------------
#

# Flag for py2 and py3 compatibility to use when separate code paths are necessary
# When _PY2 is False, we assume Python 3 is in use
_PY2 = sys.version_info.major == 2

# CfgNodes can only contain a limited set of valid types
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
# py2 allow for str and unicode
if _PY2:
    _VALID_TYPES = _VALID_TYPES.union({unicode})  # noqa: F821

# Utilities for importing modules from file paths
if _PY2:
    # imp is available in both py2 and py3 for now, but is deprecated in py3
    import imp
else:
    import importlib.util

logger = logging.getLogger(__name__)



#===== AlgConfig =====
class AlgConfig(CfgNode):

  #=============================== __init__ ==============================
  #
  def __init__(self, init_dict=None, key_list=None, new_allowed=True):

    super(AlgConfig,self).__init__(init_dict, key_list, new_allowed)

  
  #==================== _create_config_tree_from_dict ====================
  #
  @classmethod
  def _create_config_tree_from_dict(cls, dic, key_list):
    """!
    @brief  Create a configuration tree using the given dict.
            Any dict-like objects inside dict BECOME CfgNode. 

    It is up to the actual AlgConfig instance to properly parse dict-like
    objects and handle "recursion."  These most likely belong to other algorithms that
    will have their own particular structure, as opposed to a nesting of the same
    AlgConfig node type.  Thus, it gets converted to a CfgNode, which allowed
    recursion of self and will handle nested dictionaries correctly.

    Put differently, there is no recursion of derived classes allowed.  
    Dictionary elements are other configuration elements associated to
    sub-components of the algorithm deing described. They might even be
    other algorithm descriptions, in which case it is up to the derived
    class setup to properly handle the nested algorithm configurations.

    @note   Hope this works.  If so, then remove this note. DEBUG.
    @note   It does work, but how to have top-level Algorithm manage its
            sub-algorithms is still to be determined.  Until a successful
            example is done, this will be considered still open.
    @todo   Need to clean up code comments when the notes have been address.

    @param[in] dic      (dict):
    @param[in] key_list (list[str]): a list of names which index this CfgNode from the
                        root.  Currently only used for logging purposes.
    """
    dic = copy.deepcopy(dic)
    for k, v in dic.items():
      if isinstance(v, dict):
          # Convert dict to CfgNode
          dic[k] = CfgNode(v, key_list=key_list + [k])
      else:
          # Check for valid leaf type or nested CfgNode
          _assert_with_logging(
                _valid_type(v, allow_cfg_node=False),
                "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list + [k]), type(v), _VALID_TYPES
                ),
                )
    return dic

  def dump(self, **kwargs):
            """Dump to a string."""
    
            def convert_to_dict(cfg_node, key_list):
                if not isinstance(cfg_node, CfgNode):
                    _assert_with_logging(
                        _valid_type(cfg_node),
                        "Key {} with value {} is not a valid type; valid types: {}".format(
                            ".".join(key_list), type(cfg_node), _VALID_TYPES
                        ),
                    )
                    return cfg_node
                else:
                    cfg_dict = dict(cfg_node)
                    for k, v in cfg_dict.items():
                        cfg_dict[k] = convert_to_dict(v, key_list + [k])
                    return cfg_dict
    
            self_as_dict = convert_to_dict(self, [])
            return yaml.safe_dump(self_as_dict, **kwargs)
    


def _assert_with_logging(cond, msg):
    if not cond:
        logger.debug(msg)
    assert cond, msg

def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (
        allow_cfg_node and isinstance(value, CfgNode)
    )
#
#=============================== Configuration ===============================
