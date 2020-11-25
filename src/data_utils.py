import re
import copy
import datetime

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def str2list(string):
    try:
        out_list = [re.sub(r'[^0-9\-\:\.\s]', '', x).strip() for x in string.split(',')]
    except AttributeError:
        out_list = []
    return out_list

def convert_to_datetime(str_dt):
    if '.' in str_dt:
        return datetime.datetime.strptime(str_dt, '%Y-%m-%d %H:%M:%S.%f')
    else:
        return datetime.datetime.strptime(str_dt, '%Y-%m-%d %H:%M:%S')

def make_token_dict(token_list):
    if None in token_list:
        token_dict = {None:0}
        token_list = [x for x in token_list if x]
        token_dict.update(dict(zip(token_list, range(1, len(token_list)+1))))
    else:
        token_dict = dict(zip(token_list, range(len(token_list))))
    return token_dict

def from_dataframe(df, uid, data_config):
    def example_to_dict(group):
        sid, df_session = group
        return {k:v if len(v)>1 and k!=uid else v[0] for k,v in df_session.to_dict('list').items()}
    
    data_list = []
    for x in list(map(example_to_dict, df.groupby(uid))):
        data_list.append({k:x[k] if k!=uid else str(x[k]) for k in data_config.keys()})
    return data_list

def update_config(data, config):
#     assert all(k in df.columns for k in config.keys())
    udata = {}

    for k,v in config.items():
#         print(k)
        assert k in data[0]
        if isinstance(data[0][k], list):
            udata[k] = unique([x for sess in data for x in sess[k]])
        else:
            udata[k] = unique([sess[k] for sess in data])
    
    new_config = {k:copy.deepcopy(v) for k,v in config.items()}
    
    for k,v in new_config.items():
        _udata = [x for x in udata[k] if x==x and x]
        if v.mode=='uid':
            pass
        elif v.mode=='categorical':
            new_config[k].token_map = make_token_dict(udata[k])
            new_config[k].size = max(list(new_config[k].token_map.values()))+1
            
        elif v.mode=='numerical':
            new_config[k].size = 1
            new_config[k].min, new_config[k].max = min(_udata), max(_udata)
 
        elif v.mode=='datetime':
            new_config[k].min, new_config[k].max = min(_udata), max(_udata)
        else:
            raise NotImplementedError
            
        if isinstance(data[0][k], list):
            new_config[k].max_len = max(len(x[k]) for x in data)
        
    return new_config 