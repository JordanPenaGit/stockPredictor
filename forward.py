#!/usr/bin/env python3
from _bootstrap import activate
activate()
import argparse,json,logging
from quant_ml.research.forward import init_protocol,capture,evaluate_capture

def main():
    p=argparse.ArgumentParser(description='Append-only prospective SHADOW forecasts. No orders or broker.')
    p.add_argument('action',choices=['init','capture','evaluate']);p.add_argument('--root',required=True)
    p.add_argument('--signal-date',help='Required only to evaluate an existing recorded signal')
    a=p.parse_args()
    if a.action=='evaluate' and not a.signal_date:p.error('evaluate requires --signal-date')
    try:
        if a.action=='init':r=init_protocol(a.root)
        elif a.action=='capture':r=capture(a.root)
        else:r=evaluate_capture(a.root,a.signal_date)
        print(json.dumps(r,indent=2));return 0
    except Exception as exc:logging.error('%s: %s',type(exc).__name__,exc);return 1
if __name__=='__main__':raise SystemExit(main())
