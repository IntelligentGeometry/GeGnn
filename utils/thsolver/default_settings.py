"""
build a dict containing all settings which can be accessed globally.

Usage: 
all settings defined in gnndist.yaml can be accessed anywhere like this:

from utils.thsolver import default_settings
default_settings.get_global_value("max_epoch")

"""

_global_dict = None

def _init(FLAGS=None):
    global _global_dict
    _global_dict = {
    #    "normal_aware_pooling": True,
     #   "num_edge_types": 7,
    }

def set_global_value(key, value):
    _global_dict[key] = value
    
def set_global_values(FLAGS):
    """set global values from the FLAGS in thsolver.solver
    """
    for each in FLAGS:
        for it in FLAGS[each]:
            _global_dict[it] = FLAGS[each][it]
    
def get_global_value(key):
    return _global_dict[key]