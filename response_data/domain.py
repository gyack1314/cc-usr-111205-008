"""清点两段多通道资料的有效值。"""
from .io import number,positive,integer

def summarize(v):
    channels=set(v['channels']);counts={};integer(v['run_limit'],1)
    for c in v['channels'].values():number(c['center']);positive(c['scale']);positive(c['weight'])
    for name in ('reference','candidate'):
        count=0
        for row in v[name]:
            if set(row)!=channels:raise ValueError('通道引用不齐')
            for x in row.values():
                if x is not None:number(x);count+=1
        counts[name]={'rows':len(v[name]),'observed_values':count}
    return {'channels':len(channels),'series':counts}
