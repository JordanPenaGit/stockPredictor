#!/usr/bin/env python3
from _bootstrap import activate
activate()
import argparse,json,logging
from pathlib import Path
from quant_ml.research.development import run_development

def main():
    p=argparse.ArgumentParser(description='Predeclared development-only research; no old holdout reassessment')
    p.add_argument('--plan',default='audit/development-plan.json')
    p.add_argument('--output',required=True,help='New directory; existing paths refused')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--cache',help='Read-only legacy cache; slice to development boundary before modeling')
    source.add_argument('--snapshot',help='Trusted local development joblib snapshot (never load untrusted pickle)')
    p.add_argument('--smoke',action='store_true',help='First three validation months, linear only; not full research')
    a=p.parse_args()
    try:run_development(json.loads(Path(a.plan).read_text()),a.output,a.cache,a.snapshot,a.smoke)
    except Exception as exc:
        logging.error('%s: %s',type(exc).__name__,exc);return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
