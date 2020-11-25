import numpy as np
import pandas as pd

from src.data_utils import str2list, convert_to_datetime, from_dataframe, update_config
from src.featurizer import scaler, compute_wday, compute_srel
from src.parameters import Parameters

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

@dataclass
class EHRFeature:
    """
    Specify the features and types input to the first model layer
    """
    pat_id: str
    label: int
    race: int
    ethnic: int
    sex: List[int]
    marital_status: List[int]
    age: float
    srel: List[float]
    wday: List[int]
    seq_mask: List[int]

    
    def __post_init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

def convert_example_to_ehr_feature(example, config):
    feature_dict = {}
    for k,v in example.items():
        if k=='time':
            npad = config['time'].max_len-len(v)
            dt_min, dt_max = config['time'].min, config['time'].max
            srel_min, srel_max = 0.0, (dt_max-dt_min).total_seconds()
            
            feature_dict.update({'seq_mask':[1]*len(v)+[0]*npad})
            feature_dict.update(compute_srel(v, npad, dt_min, srel_min, srel_max))
            feature_dict.update(compute_wday(v, npad))
        elif k=='age':
            feature_dict[k] = scaler(v, config['age'].min, config['age'].max)
        elif config[k].token_map is not None:
            feature_dict[k] = config[k].token_map[v]
        else:
            feature_dict[k] = v

    return feature_dict

def ehr_from_dataframe(csvpath, data_config):
    df = pd.read_csv(csvpath)
    df = df.rename(columns={k:k.replace(' ', '_') for k in df.columns})
    df = df.replace({np.nan:None})
    
    print('loading dataframe')
    data_list = from_dataframe(df, 'pat_id', data_config,)
    
    print('converting datetimes')
    for data in data_list:
        data['time'] = list(map(convert_to_datetime, str2list(data['time'])))
        
    print('updating config')
    config = update_config(data_list, data_config)
    
    print('creating features')
    feature_list = list(map(lambda x: convert_example_to_ehr_feature(x, config), data_list))
    config['srel'] = Parameters(mode='numerical', vector='linear')
    config['wday'] = Parameters(mode='categorical', vector='embedding')
    _ = config.pop('time')
    config = update_config(feature_list, config)
    
    for x in feature_list:
        for k in ['wday']:
            x[k] = list(map(config[k].token_map.get, x[k]))
            
    features = []
    for x in feature_list:
        features.append(EHRFeature(**x))
    return features, config
 