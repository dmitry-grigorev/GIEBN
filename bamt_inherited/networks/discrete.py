from bamt_inherited.networks.base import BaseNetworkGI
from bamt.networks import discrete_bn as DiscreteBN

from typing import Dict, Tuple, List, Callable, Optional, Type, Union, Any, Sequence

class DiscreteBNGI(BaseNetworkGI):
    def __init__(self, outputdirectory: str, random_state=42, max_cat=3, custom_mapper: Dict[str, Dict[int, str]]=None):
        BaseNetworkGI.__init__(self, outputdirectory, random_state, max_cat, custom_mapper)
        self.type = "Discrete"
        self.scoring_function = ""
        self._allowed_dtypes = ["disc", "disc_num"]
        self.has_logit = None
        self.use_mixture = None
