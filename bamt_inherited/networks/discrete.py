from bamt_inherited.networks.base import BaseNetworkGI
from bamt.networks import DiscreteBN

from typing import Dict, Tuple, List, Callable, Optional, Type, Union, Any, Sequence

class DiscreteBNGI(BaseNetworkGI, DiscreteBN):
    def __init__(self, outputdirectory: str, random_state=42, max_cat=3, custom_mapper: Dict[str, Dict[int, str]]=None):
        BaseNetworkGI.__init__(self, outputdirectory, random_state, max_cat, custom_mapper)
        DiscreteBN.__init__(self)
