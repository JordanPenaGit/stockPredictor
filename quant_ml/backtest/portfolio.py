"""Self-financing fractional adjusted-share ledger, proportional all-in costs."""
from dataclasses import dataclass, field
import math
@dataclass
class Portfolio:
    cash: float
    quantities: dict = field(default_factory=dict)
    basis: dict = field(default_factory=dict)
    trades: list = field(default_factory=list)
    realized: list = field(default_factory=list)
    def equity(self,prices):
        for s,q in self.quantities.items():
            if q and (s not in prices or not math.isfinite(prices[s]) or prices[s]<=0):raise ValueError(f'Missing valuation for held {s}; stopping instead of inventing a price')
        return self.cash+sum(q*prices[s] for s,q in self.quantities.items() if q)
    def rebalance(self,weights,prices,cost_rate,date):
        if any(not math.isfinite(w) or w<0 for w in weights.values()) or sum(weights.values())>1+1e-9:raise ValueError('Finite long-only unlevered weights required')
        if not math.isfinite(cost_rate) or not 0 <= cost_rate < 1:raise ValueError('Cost rate must be finite and in [0, 1)')
        pre=self.equity(prices);total_cost=0;notional=0
        if not math.isfinite(pre) or pre<=0:raise ValueError('Portfolio equity must be finite and positive')
        # Solve target post-fee NAV; missing opens are never filled. Existing missing holding fails valuation.
        valid={s:w for s,w in weights.items() if s in prices and math.isfinite(prices[s]) and prices[s]>0}
        # Bisection is stable even for large admissible fees, unlike fixed-point iteration.
        holdings={s:q*prices[s] for s,q in self.quantities.items() if q}
        symbols=set(valid)|set(holdings)
        low,high=0.0,pre
        for _ in range(80):
            target_nav=(low+high)/2
            turnover=sum(abs(valid.get(s,0)*target_nav-holdings.get(s,0)) for s in symbols)
            if target_nav+cost_rate*turnover>pre:high=target_nav
            else:low=target_nav
        target_nav=(low+high)/2
        if target_nav<=0:raise ValueError('Costs exhaust capital')
        delta={s:valid.get(s,0)*target_nav/prices[s]-self.quantities.get(s,0) for s in set(valid)|set(self.quantities) if s in prices and math.isfinite(prices[s]) and prices[s]>0}
        for s,dq in sorted(delta.items(),key=lambda item:item[1]):
            if abs(dq)<1e-10:continue
            p=prices[s];value=abs(dq)*p;fee=value*cost_rate
            oldq=self.quantities.get(s,0);oldbasis=self.basis.get(s,0)
            if dq<0:
                entry=oldbasis*min(-dq/oldq,1);pnl=value-fee-entry
                self.realized.append({'date':date,'ticker':s,'pnl':pnl,'return':pnl/entry if entry else 0,'cost_basis':entry})
                self.basis[s]=max(0,oldbasis-entry)
            else:self.basis[s]=oldbasis+value+fee
            self.quantities[s]=max(0,oldq+dq);self.cash-=dq*p+fee
            self.trades.append({'date':date,'ticker':s,'quantity':dq,'price':p,'notional':value,'cost':fee,'cash_after':self.cash})
            total_cost+=fee;notional+=value
        if self.cash < -1e-6:raise AssertionError('Negative cash')
        self.cash=max(0,self.cash)
        assert abs(self.equity(prices)-(pre-total_cost))<max(1e-5,pre*1e-10)
        return notional/pre,total_cost
