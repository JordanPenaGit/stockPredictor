import argparse,logging,re
from .config import load_config,ROOT
from .workflow import train_research,assess,resolve_run

def main(command):
    parser=argparse.ArgumentParser(description='Leakage-aware local quantitative research, not trade execution')
    parser.add_argument('--run',help='Frozen run ID (default: latest; training default research-v1)')
    if command=='train':
        parser.add_argument('--config',help='JSON configuration file');parser.add_argument('--refresh',action='store_true')
    elif command=='analyze':
        parser.add_argument('symbol');parser.add_argument('--cached',action='store_true',help='Use matching-date cache, no forced network refresh')
    elif command=='screen':
        parser.add_argument('--top',type=int,default=20);parser.add_argument('--tickers',nargs='+');parser.add_argument('--cached',action='store_true')
    args=parser.parse_args()
    if command=='screen' and args.top<1:parser.error('--top must be positive')
    if command in ['analyze','screen']:
        symbols=[args.symbol] if command=='analyze' else args.tickers
        if symbols is not None:
            symbols=[s.strip().upper() for s in symbols]
            if any(not re.fullmatch(r'[A-Z0-9^][A-Z0-9.^=-]{0,19}',s) for s in symbols):
                parser.error('Invalid ticker format')
            symbols=list(dict.fromkeys(symbols))
    if args.run:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',args.run):parser.error('Run ID must be a simple name, not a path')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        if command=='train':train_research(load_config(args.config),args.run or 'research-v1',args.refresh)
        elif command=='backtest':assess(resolve_run(args.run))
        else:
            from .models.predict import current_predictions,print_analysis
            table,c=current_predictions(args.run,symbols,refresh=not args.cached)
            if command=='analyze':print_analysis(table,c)
            else:
                display=table.head(args.top)[['ticker','predicted_return','probability','score','signal']].copy()
                print(display.to_string(formatters={'predicted_return':'{:+.2%}'.format,'probability':'{:.1%}'.format,'score':'{:.4f}'.format}))
                print('Screen signal labels are descriptive, not the backtested portfolio selection rule.')
                print(f'As of {table.date.iloc[0].date()}; {c.horizon}-session next-open horizon. Model estimates, not guarantees. Universe {len(table)}; requested top {args.top}.')
        return 0
    except Exception as exc:
        logging.error('%s: %s',type(exc).__name__,exc);return 1
