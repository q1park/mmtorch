import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

@dataclass
class Parameters:
    """
    Specify the features and types input to the first model layer
    """
    mode: str
    vector: Union[str, Tuple[str]]
    sub_mode: Optional[str] = None
    token_map: Optional[dict] = None
    
    size: Optional[int] = None
    min: Optional[Union[int, float, str]] = None
    max: Optional[Union[int, float, str]] = None
    range: Optional[Tuple[Union[int, float]]] = None
    max_len: Optional[int] = None
        
    d_layer_1: Optional[int] = None
    d_layer_2: Optional[int] = None
    d_layer_3: Optional[int] = None
    d_layer_4: Optional[int] = None
        
        
