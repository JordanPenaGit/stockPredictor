from pathlib import Path
import json
from dataclasses import replace
from quant_ml.config import Config,ROOT
from quant_ml.data.downloader import MarketData
from quant_ml.features import make_dataset
from quant_ml.models.train import walk_forward,evaluate
from quant_ml.backtest.engine import simulate
from quant_ml.backtest.metrics import performance
c=replace(Config(),models=['linear']);data=MarketData().universe(c);p,features=make_dataset(data,c)
pred,audit=walk_forward(p,features,'linear',c,'2020-01-01','2021-01-01')
curve,trades,realized=simulate(data,pred,c)
root=ROOT/'results'/'foundation';root.mkdir(parents=True,exist_ok=True)
summary={'stage':'real-data baseline foundation verification','symbols':{k:len(v) for k,v in data.items()},'features':len(features),'rows':len(p),'folds':audit,'model_metrics':evaluate(pred,'2021-01-01'),'portfolio':performance(curve,trades,realized,c.starting_capital)}
(root/'verification.json').write_text(json.dumps(summary,indent=2));pred.to_csv(root/'predictions.csv',index=False);curve.to_csv(root/'equity.csv');print(json.dumps(summary,indent=2))
