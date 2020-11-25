import numpy as np
import pandas as pd

from src.data_utils import str2list, convert_to_datetime, from_dataframe, update_config
from src.featurizer import scaler, compute_wtime, compute_wday, compute_month, compute_day
from src.parameters import Parameters

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

@dataclass(frozen=True)
class YCFeature:
    """
    Specify the features and types input to the first model layer
    """
    sid: str
    label: int
    month: List[int]
    day: List[int]
    wday: List[int]
    oddtime: List[float]
    eventime: List[float]
    category: List[int]
    duration: List[float]
    cid: List[int]
    nsessbuy: List[int]
    nbuysess: List[float]
    price: List[float]
    totbuys: List[int]
    totclicks: List[int]
    totdurs: List[float]
    seq_mask: List[int]
    buy_datetime: Optional[str] = None
    buy_cid: Optional[str] = None
    buy_price: Optional[float] = None
    buy_quant: Optional[int] = None
        
    def __post_init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

def aggregate_sessions(users, items):
    item_dict = items.set_index('cid').T.to_dict('list')
    sessions = pd.DataFrame(list(map(lambda x: [x]+item_dict.get(x), users['cid'])), columns = items.keys())
    
    return pd.concat([users.drop(columns=['cid']), sessions], axis=1)

def convert_example_to_yc_feature(example, config):
    feature_dict = {}
    for k,v in example.items():
        if k=='sid':
            feature_dict[k] = str(v)
        elif k=='label':
            feature_dict[k] = v
        else:
            npad = config['datetime'].max_len-len(v)
            feature_dict.update({'seq_mask':[1]*len(v)+[0]*npad})
            if k=='datetime':
                npad = config['datetime'].max_len-len(v)
                feature_dict.update(compute_wtime(v, npad))
                feature_dict.update(compute_wday(v, npad))
                feature_dict.update(compute_month(v, npad))
                feature_dict.update(compute_day(v, npad))
            elif config[k].mode=='numerical':
                feature_dict[k] = list(map(lambda x: scaler(x, config['duration'].min, config['duration'].max), v))+[0]*npad
            elif config[k].token_map is not None:
                feature_dict[k] = list(map(config[k].token_map.get, v))+[0]*npad
            else:
                feature_dict[k] = v+[0]*npad

    return feature_dict

def yc_from_dataframe(csv_user, csv_item, csv_target, data_config):
#     assert all(k in df.columns for k in config.keys())
    df_user = pd.read_csv(csv_user, names=['sid', 'datetime', 'cid', 'category', 'duration'])
    df_item = pd.read_csv(csv_item, names=['cid', 'nsessbuy', 'nbuysess', 'price', 'totbuys', 'totclicks', 'totdurs'])
    df_target = pd.read_csv(csv_target, names=['sid', 'datetime', 'cid', 'price', 'quantity'])
    
    print('loading dataframe')
    df = aggregate_sessions(df_user, df_item)
    df = df.replace({np.nan:None})
    data_list = from_dataframe(df, 'sid', data_config)
    
    print('adding labels and datetimes')
    label_dict = {str(k):v['quantity'].sum() for k,v in df_target.groupby('sid')}
    label_dict = {k:v if v==0 else 1 for k,v in label_dict.items()}
    
    for data in data_list:
        data['label'] = label_dict[data['sid']]
        data['datetime'] = list(map(convert_to_datetime, data['datetime']))
        
    print('updating config')
    data_config['label'] = Parameters(mode='categorical', vector=None)
    config = update_config(data_list, data_config)
    
    print('creating features')
    feature_list = list(map(lambda x: convert_example_to_yc_feature(x, config), data_list))
    config['month'] = Parameters(mode='categorical', vector='embedding')
    config['day'] = Parameters(mode='categorical', vector='embedding')
    config['wday'] = Parameters(mode='categorical', vector='embedding')
    config['oddtime'] = Parameters(mode='numerical', vector='linear')
    config['eventime'] = Parameters(mode='numerical', vector='linear')
    _ = config.pop('datetime')
    config = update_config(feature_list, config)
    
    for x in feature_list:
        for k in ['month', 'day', 'wday']:
            x[k] = list(map(config[k].token_map.get, x[k]))
    
    features = []
    for x in feature_list:
        features.append(YCFeature(**x))
    return features, config
 