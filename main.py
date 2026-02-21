"""
main.py
=======
Polymarket Copy Trader — Paper Mode
  ▸ Copy trader   : follows target wallet at 50% scale
  ▸ Sniper tester : paper-trades the volume-spike signal (separate budget)
  ▸ Data collector: snapshots every 5-min BTC market at key timestamps
"""

import asyncio, json, logging, os, re, sys, time, threading, csv
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
load_dotenv()

def load_config(path="config.yaml"):
    try:
        with open(path) as f: return yaml.safe_load(f)
    except FileNotFoundError: return {}

cfg      = load_config()
LOG_FILE = cfg.get("log_file", "logs/bot.log")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_WALLET    = "0x63ce342161250d705dc0b16df89036c8e5f9ba9a"
DATA_API         = "https://data-api.polymarket.com"
GAMMA_API        = "https://gamma-api.polymarket.com"
STARTING_BUDGET  = 100.0
COPY_SCALE       = 0.5
POLL_INTERVAL    = 5
MEMORY_FILE      = "logs/trader_memory.json"
COLLECTOR_FILE   = "logs/live_market_data.csv"
SNIPER_FILE      = "logs/sniper_trades.csv"
SNIPER_BUDGET    = 100.0        # separate paper budget for sniper
SPIKE_VOL_RATIO  = 10.0         # vol_now / vol_baseline to trigger
SPIKE_MOVE       = 0.12         # price move > 12% to trigger
SNIPER_BET_SIZE  = 10.0

# ── COLORS ────────────────────────────────────────────────────────────────────
class C:
    R="\033[0m";B="\033[1m";CY="\033[96m";MG="\033[95m"
    GR="\033[92m";RE="\033[91m";YL="\033[93m";BL="\033[94m"
    WH="\033[97m";GY="\033[90m";OR="\033[38;5;208m"

def _c(t,c):   return f"{c}{t}{C.R}"
def green(t):  return _c(t,C.GR)
def red(t):    return _c(t,C.RE)
def yel(t):    return _c(t,C.YL)
def cyan(t):   return _c(t,C.CY)
def gray(t):   return _c(t,C.GY)
def bold(t):   return _c(t,C.B)
def blue(t):   return _c(t,C.BL)
def orange(t): return _c(t,C.OR)
def mg(t):     return _c(t,C.MG)
def pnlc(v,t): return green(t) if v>=0 else red(t)
def trunc(t,n):return t[:n-2]+".." if len(t)>n else t

def strip_ansi(t):
    return re.sub(r'\033\[[0-9;]*m','',t)

def pad(t,n,align='<'):
    raw=strip_ansi(t); extra=len(t)-len(raw); w=n+extra
    if align=='>': return t.rjust(w)
    if align=='^': return t.center(w)
    return t.ljust(w)

# ── TIME HELPERS ──────────────────────────────────────────────────────────────

def slug_to_ts(slug:str)->int:
    m=re.search(r'-(\d{9,11})$',slug)
    return int(m.group(1)) if m else 0

def ts_to_iso(ts:int)->str:
    return datetime.fromtimestamp(ts,tz=timezone.utc).isoformat()

def time_left_from_ts(end_ts:int):
    """Returns (colored_str, seconds_left). Uses raw unix ts — always accurate."""
    secs=int(end_ts - time.time())
    if secs<=0: return red("ENDED"),-1
    h,rem=divmod(secs,3600); m,s=divmod(rem,60)
    if h>0:   return yel(f"{h}h {m:02d}m"),secs
    elif m>0: return orange(f"{m}m {s:02d}s"),secs
    else:     return red(f"{s}s!!"),secs

def time_left(end_str:str):
    """Wrapper: parse ISO string then use ts-based calc."""
    if not end_str: return gray("Unknown"),0
    try:
        end=datetime.fromisoformat(end_str.replace('Z','+00:00'))
        return time_left_from_ts(int(end.timestamp()))
    except: return gray("Unknown"),0

def end_date_from_slug(slug:str)->str:
    ts=slug_to_ts(slug)
    return ts_to_iso(ts) if ts else ""

def get_end_date(slug:str,raw_end:str)->str:
    if raw_end: return raw_end
    return end_date_from_slug(slug)

# ── MEMORY ────────────────────────────────────────────────────────────────────

def load_memory()->dict:
    os.makedirs("logs",exist_ok=True)
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE) as f: data=json.load(f)
            bal=data.get('balance',STARTING_BUDGET)
            n=len(data.get('closed_trades',[]))
            print(f"  {green('Memory loaded')}  balance={cyan(f'${bal:.2f}')}  history={yel(str(n))} trades")
            return data
        except Exception as e:
            print(f"  {red('Memory error')}: {e}")
    return {}

def save_memory(m:dict):
    try:
        with open(MEMORY_FILE,'w') as f: json.dump(m,f,indent=2)
    except Exception as e: logging.getLogger("memory").error(f"Save: {e}")

# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    key:str; market_title:str; outcome:str; slug:str; condition_id:str
    entry_price:float; cur_price:float; shares:float
    entry_amount:float; cur_value:float; end_date:str
    opened_at:float=field(default_factory=time.time)

    @property
    def end_ts(self)->int:
        ts=slug_to_ts(self.slug)
        if ts: return ts
        if self.end_date:
            try: return int(datetime.fromisoformat(self.end_date.replace('Z','+00:00')).timestamp())
            except: pass
        return 0

    @property
    def url(self): return f"https://polymarket.com/event/{self.slug}" if self.slug else "N/A"
    @property
    def pnl(self): return self.cur_value-self.entry_amount
    @property
    def roi_pct(self):
        return (self.cur_price-self.entry_price)/self.entry_price*100 if self.entry_price>0 else 0.0

@dataclass
class ClosedTrade:
    key:str; market_title:str; outcome:str; slug:str
    entry_price:float; exit_price:float; entry_amount:float; exit_amount:float
    realized_pnl:float; closed_at:float=field(default_factory=time.time); reason:str="closed"

    @property
    def url(self): return f"https://polymarket.com/event/{self.slug}" if self.slug else "N/A"
    @property
    def roi_pct(self):
        return (self.exit_price-self.entry_price)/self.entry_price*100 if self.entry_price>0 else 0.0

    def to_dict(self):
        return {k:getattr(self,k) for k in
                ['key','market_title','outcome','slug','entry_price','exit_price',
                 'entry_amount','exit_amount','realized_pnl','closed_at','reason']}
    @classmethod
    def from_dict(cls,d):
        keys=['key','market_title','outcome','slug','entry_price','exit_price',
              'entry_amount','exit_amount','realized_pnl','closed_at','reason']
        return cls(**{k:d[k] for k in keys if k in d})

# ── COPY TRADER ───────────────────────────────────────────────────────────────

class CopyTrader:
    def __init__(self,wallet:str,memory:dict):
        self.wallet=wallet; self.scale=COPY_SCALE
        self.available=memory.get('balance',STARTING_BUDGET)
        self.invested=memory.get('invested',0.0)
        self.returned=memory.get('returned',0.0)
        self.realized=memory.get('realized',0.0)
        self.session_start=self.available
        self.closed_trades:List[ClosedTrade]=[ClosedTrade.from_dict(d) for d in memory.get('closed_trades',[])]
        self.positions:Dict[str,Position]={}
        self._prev_target:Dict[str,dict]={}
        self.manual_queue:List[str]=[]
        self.session=requests.Session()
        self.session.headers.update({'Accept':'application/json','User-Agent':'Mozilla/5.0'})
        self.logger=logging.getLogger("trader")

    @property
    def budget(self): return self.available+self.invested

    def to_memory(self):
        return {'balance':self.available,'invested':self.invested,'returned':self.returned,
                'realized':self.realized,'closed_trades':[t.to_dict() for t in self.closed_trades[-500:]],
                'saved_at':datetime.now().isoformat()}

    def _fetch_raw(self)->Dict[str,dict]:
        try:
            r=self.session.get(f"{DATA_API}/positions",
                params={"user":self.wallet,"sizeThreshold":0.01,"limit":500},timeout=15)
            r.raise_for_status()
            out={}
            for p in r.json():
                cid=p.get('conditionId',''); oc=p.get('outcome','Unknown'); key=f"{cid}_{oc}"
                raw_end=(p.get('endDate') or p.get('end_date') or p.get('endDateIso') or p.get('expirationDate') or '')
                slug=p.get('slug','')
                out[key]={'condition_id':cid,'market_title':p.get('title','Unknown'),'outcome':oc,
                          'avg_price':float(p.get('avgPrice',0) or 0),'cur_price':float(p.get('curPrice',0) or 0),
                          'shares':float(p.get('size',0) or 0),'current_value':float(p.get('currentValue',0) or 0),
                          'slug':slug,'end_date':get_end_date(slug,raw_end)}
            return out
        except Exception as e:
            self.logger.error(f"Fetch: {e}"); return self._prev_target

    def _do_close(self,key:str,reason:str="closed")->Optional[ClosedTrade]:
        if key not in self.positions: return None
        pos=self.positions[key]
        exit_val=pos.cur_price*pos.shares; profit=exit_val-pos.entry_amount
        self.available+=exit_val; self.invested=max(0.0,self.invested-pos.entry_amount)
        self.returned+=exit_val; self.realized+=profit
        ct=ClosedTrade(key=key,market_title=pos.market_title,outcome=pos.outcome,slug=pos.slug,
                       entry_price=pos.entry_price,exit_price=pos.cur_price,entry_amount=pos.entry_amount,
                       exit_amount=exit_val,realized_pnl=profit,reason=reason)
        self.closed_trades.append(ct); del self.positions[key]; return ct

    def close_all(self,reason:str="shutdown")->List[ClosedTrade]:
        return [ct for key in list(self.positions.keys()) for ct in [self._do_close(key,reason)] if ct]

    def sync(self)->List[tuple]:
        events=[]; current=self._fetch_raw()
        for key,raw in current.items():
            if key not in self._prev_target:
                needed=raw['avg_price']*raw['shares']*self.scale
                if needed<0.01: continue
                if needed>self.available:
                    events.append(("skip",f"SKIP {trunc(raw['market_title'],35)} | Need ${needed:.2f}, have ${self.available:.2f}")); continue
                our_shares=raw['shares']*self.scale
                self.available-=needed; self.invested+=needed
                pos=Position(key=key,market_title=raw['market_title'],outcome=raw['outcome'],slug=raw['slug'],
                             condition_id=raw['condition_id'],entry_price=raw['avg_price'],cur_price=raw['cur_price'],
                             shares=our_shares,entry_amount=needed,cur_value=raw['cur_price']*our_shares,
                             end_date=raw['end_date'])
                self.positions[key]=pos
                tl,_=time_left(pos.end_date)
                events.append(("new",f"NEW BET  {raw['market_title']}\n"
                               f"         Side={raw['outcome']}  Entry=${raw['avg_price']:.4f}  "
                               f"Invested=${needed:.2f} ({our_shares:.1f} shares)\n"
                               f"         Time Left={tl}\n         URL={pos.url}"))
                self.logger.info(f"COPIED: {raw['market_title']} | {raw['outcome']} | ${needed:.2f}")

        for key,pos in list(self.positions.items()):
            if key in current:
                pos.cur_price=current[key]['cur_price']
                pos.cur_value=pos.cur_price*pos.shares
                _,secs=time_left_from_ts(pos.end_ts)
                if secs==-1:
                    ct=self._do_close(key,reason="expired")
                    if ct: events.append(("close",f"EXPIRED  {pos.market_title}\n"
                                          f"         Profit=${ct.realized_pnl:+.2f}  ROI={ct.roi_pct:+.1f}%"))
            elif key in self._prev_target:
                ct=self._do_close(key,reason="closed")
                if ct: events.append(("close",f"TARGET CLOSED  {pos.market_title}\n"
                                      f"               Profit=${ct.realized_pnl:+.2f}  ROI={ct.roi_pct:+.1f}%"))

        for key in list(self.manual_queue):
            if key in self.positions:
                pos=self.positions[key]; ct=self._do_close(key,reason="manual")
                if ct: events.append(("close",f"MANUAL CLOSE  {pos.market_title}\n"
                                      f"              Profit=${ct.realized_pnl:+.2f}  ROI={ct.roi_pct:+.1f}%"))
        self.manual_queue.clear()
        self._prev_target=current; return events

    def summary(self):
        unreal=sum(p.pnl for p in self.positions.values())
        return {'budget':self.budget,'available':self.available,'invested':self.invested,
                'returned':self.returned,'realized':self.realized,'unrealized':unreal,
                'value':self.available+sum(p.cur_value for p in self.positions.values()),
                'n_open':len(self.positions),'n_closed':len(self.closed_trades),
                'session_start':self.session_start}

# ── SNIPER TESTER ─────────────────────────────────────────────────────────────

@dataclass
class SniperTrade:
    slug:str; direction:str; entry_price:float; bet_size:float
    vol_ratio:float; move:float; entered_at:float=field(default_factory=time.time)
    outcome:str=""; exit_price:float=0.0; profit:float=0.0; status:str="open"

    @property
    def end_ts(self)->int: return slug_to_ts(self.slug)
    @property
    def url(self): return f"https://polymarket.com/event/{self.slug}"
    @property
    def roi_pct(self): return self.profit/self.bet_size*100 if self.bet_size>0 else 0.0

class SniperTester:
    """
    Paper-trades the volume-spike signal independently from copy trader.
    Signal: when vol spikes >10x AND price moves >12% in last 30s,
            follow the price direction.
    """
    def __init__(self):
        self.capital=SNIPER_BUDGET
        self.session_start=SNIPER_BUDGET
        self.open_trades:Dict[str,SniperTrade]={}
        self.closed_trades:List[SniperTrade]=[]
        self._baseline_vols:Dict[str,float]={}   # slug -> vol at 240s
        self._baseline_prices:Dict[str,float]={} # slug -> price at 240s
        self._fired:set=set()                     # slugs already traded
        self.logger=logging.getLogger("sniper")
        self._ensure_csv()

    def _ensure_csv(self):
        os.makedirs("logs",exist_ok=True)
        if not os.path.exists(SNIPER_FILE):
            with open(SNIPER_FILE,'w',newline='',encoding='utf-8') as f:
                csv.DictWriter(f,fieldnames=['time','slug','direction','entry_price','exit_price',
                    'bet_size','profit','roi_pct','vol_ratio','move','outcome','status']).writeheader()

    def _append_csv(self,t:SniperTrade):
        with open(SNIPER_FILE,'a',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=['time','slug','direction','entry_price','exit_price',
                'bet_size','profit','roi_pct','vol_ratio','move','outcome','status']).writerow({
                'time':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'slug':t.slug,'direction':t.direction,'entry_price':round(t.entry_price,4),
                'exit_price':round(t.exit_price,4),'bet_size':t.bet_size,
                'profit':round(t.profit,4),'roi_pct':round(t.roi_pct,2),
                'vol_ratio':round(t.vol_ratio,1),'move':round(t.move,3),
                'outcome':t.outcome,'status':t.status})

    def on_snapshot(self,slug:str,up_price:float,volume:float,seconds_before:int):
        """Called by collector each time a snapshot is recorded."""
        # Store baseline at 240s
        if seconds_before==240:
            self._baseline_vols[slug]=volume
            self._baseline_prices[slug]=up_price

        # Check for spike at 30s or 10s
        if seconds_before not in (30,10): return
        if slug in self._fired: return
        if up_price<=0.01 or up_price>=0.99: return

        base_vol   = self._baseline_vols.get(slug,0)
        base_price = self._baseline_prices.get(slug,up_price)
        if base_vol<=0: return

        vol_ratio = volume/base_vol
        move      = up_price-base_price

        if vol_ratio < SPIKE_VOL_RATIO: return
        if abs(move)  < SPIKE_MOVE:    return

        # SIGNAL FIRED
        direction = "UP" if move>0 else "DOWN"
        entry     = up_price if move>0 else 1.0-up_price
        if self.capital < SNIPER_BET_SIZE: return

        shares    = SNIPER_BET_SIZE / entry
        self.capital -= SNIPER_BET_SIZE
        trade = SniperTrade(slug=slug,direction=direction,entry_price=entry,
                            bet_size=SNIPER_BET_SIZE,vol_ratio=vol_ratio,move=move)
        self.open_trades[slug]=trade
        self._fired.add(slug)
        self.logger.info(f"SNIPER ENTER: {slug[-10:]} {direction} entry={entry:.3f} "
                         f"vol={vol_ratio:.0f}x move={move:+.2f}")

    def on_outcome(self,slug:str,outcome:str):
        """Called when a market resolves."""
        if slug not in self.open_trades: return
        t=self.open_trades.pop(slug)
        t.outcome=outcome
        won = (t.direction==outcome)
        t.exit_price = 1.0 if won else 0.0
        t.profit     = (SNIPER_BET_SIZE/t.entry_price - SNIPER_BET_SIZE) if won else -SNIPER_BET_SIZE
        t.status     = "won" if won else "lost"
        self.capital += SNIPER_BET_SIZE/t.entry_price if won else 0
        self.closed_trades.append(t)
        self._append_csv(t)
        self.logger.info(f"SNIPER CLOSE: {slug[-10:]} {outcome} {'WIN' if won else 'LOSS'} "
                         f"profit={t.profit:+.2f}")

    def summary(self):
        n=len(self.closed_trades)
        won=sum(1 for t in self.closed_trades if t.status=="won")
        pnl=sum(t.profit for t in self.closed_trades)
        return {'capital':self.capital,'n_open':len(self.open_trades),
                'n_closed':n,'n_won':won,'pnl':pnl,
                'win_rate':won/n if n else 0,
                'session_start':self.session_start}

# ── DATA COLLECTOR ────────────────────────────────────────────────────────────

COLLECTOR_SNAPS=[240,180,120,60,30,10]

def _ensure_collector_csv():
    os.makedirs("logs",exist_ok=True)
    if not os.path.exists(COLLECTOR_FILE):
        with open(COLLECTOR_FILE,'w',newline='',encoding='utf-8') as f:
            csv.DictWriter(f,fieldnames=['recorded_at','slug','end_ts','seconds_before_close',
                                         'up_price','volume','outcome']).writeheader()

def _append_collector_row(row:dict):
    with open(COLLECTOR_FILE,'a',newline='',encoding='utf-8') as f:
        csv.DictWriter(f,fieldnames=list(row.keys())).writerow(row)

def _fetch_market(slug:str)->tuple:
    """Returns (up_price, outcome, volume)."""
    try:
        r=requests.get(f"{GAMMA_API}/markets",params={"slug":slug},timeout=8,
                       headers={"User-Agent":"Mozilla/5.0"})
        data=r.json()
        if not data or not isinstance(data,list): return None,None,0.0
        m=data[0]
        prices=m.get('outcomePrices',[]); outcomes=m.get('outcomes',[])
        if isinstance(prices,str): prices=json.loads(prices)
        if isinstance(outcomes,str): outcomes=json.loads(outcomes)
        if len(prices)<2 or len(outcomes)<2: return None,None,0.0
        p0,p1=float(prices[0]),float(prices[1])
        outcome=None
        if abs(p0-1.0)<0.01: outcome='UP' if 'up' in str(outcomes[0]).lower() else 'DOWN'
        elif abs(p1-1.0)<0.01: outcome='UP' if 'up' in str(outcomes[1]).lower() else 'DOWN'
        up_idx=next((i for i,o in enumerate(outcomes) if 'up' in str(o).lower()),0)
        up_p=float(prices[up_idx])
        up_price=up_p if 0.01<up_p<0.99 else None
        volume=float(m.get('volumeNum',0) or 0)
        return up_price,outcome,volume
    except: return None,None,0.0

class _MarketTracker:
    def __init__(self,slug:str):
        self.slug=slug; self.end_ts=slug_to_ts(slug)
        self.recorded:set=set(); self.outcome:Optional[str]=None

    def secs_left(self)->int: return max(0,self.end_ts-int(time.time()))
    def expired(self)->bool:  return time.time()>self.end_ts+90

    def tick(self,sniper:SniperTester,logger)->Optional[str]:
        secs=self.secs_left()
        target=next((s for s in COLLECTOR_SNAPS if secs<=s+15 and s not in self.recorded),None)
        if target is None: return None
        up_price,outcome,volume=_fetch_market(self.slug)
        if up_price is not None or outcome is not None:
            row={'recorded_at':datetime.now(timezone.utc).isoformat(),'slug':self.slug,
                 'end_ts':self.end_ts,'seconds_before_close':target,
                 'up_price':round(up_price,4) if up_price else '','volume':round(volume,2),
                 'outcome':outcome or ''}
            _append_collector_row(row)
            self.recorded.add(target)
            # Feed sniper
            if up_price: sniper.on_snapshot(self.slug,up_price,volume,target)
        if outcome and not self.outcome:
            self.outcome=outcome
            sniper.on_outcome(self.slug,outcome)
            return f"RESOLVED {self.slug[-10:]} → {outcome}"
        return None

async def _run_collector(sniper:SniperTester,logger):
    _ensure_collector_csv()
    trackers:Dict[str,_MarketTracker]={}
    while True:
        try:
            now=int(time.time()); cur=(now//300)*300
            for i in range(4):
                slug=f"btc-updown-5m-{cur+i*300}"
                if slug not in trackers: trackers[slug]=_MarketTracker(slug)
            for slug,tr in list(trackers.items()):
                tr.tick(sniper,logger)
                if tr.expired(): del trackers[slug]
        except Exception as e: logger.error(f"Collector: {e}")
        await asyncio.sleep(20)

# ── DASHBOARD ─────────────────────────────────────────────────────────────────

W=122

DASHBOARD_JSON = "dashboard_data.json"

def write_dashboard_json(trader:CopyTrader, sniper:SniperTester,
                          start_time:float, alerts:List[str]):
    """Writes a JSON snapshot read by dashboard.html every 3s."""
    s  = trader.summary()
    ss = sniper.summary()
    ps = sorted(trader.positions.values(), key=lambda p: p.entry_amount, reverse=True)

    invested_pct = s['invested'] / s['budget'] * 100 if s['budget'] else 0
    n_rows = sum(1 for _ in open(COLLECTOR_FILE)) - 1 if os.path.exists(COLLECTOR_FILE) else 0

    data = {
        "wallet":     TARGET_WALLET,
        "start_time": start_time,
        "updated_at": time.time(),
        "summary": {
            "budget":       round(s['budget'],    2),
            "available":    round(s['available'], 2),
            "invested":     round(s['invested'],  2),
            "invested_pct": round(invested_pct,   1),
            "realized":     round(s['realized'],  2),
            "unrealized":   round(s['unrealized'],2),
            "returned":     round(s['returned'],  2),
            "session_start":round(s['session_start'], 2),
            "n_open":       s['n_open'],
            "n_closed":     s['n_closed'],
        },
        "positions": [
            {
                "market_title": p.market_title,
                "outcome":      p.outcome,
                "slug":         p.slug,
                "entry_price":  round(p.entry_price, 4),
                "cur_price":    round(p.cur_price,   4),
                "entry_amount": round(p.entry_amount,2),
                "cur_value":    round(p.cur_value,   2),
                "end_ts":       p.end_ts,
                "url":          p.url,
            } for p in ps[:20]
        ],
        "closed_trades": [
            {
                "market_title": t.market_title,
                "outcome":      t.outcome,
                "entry_price":  round(t.entry_price,  4),
                "exit_price":   round(t.exit_price,   4),
                "entry_amount": round(t.entry_amount, 2),
                "exit_amount":  round(t.exit_amount,  2),
                "realized_pnl": round(t.realized_pnl, 2),
                "closed_at":    t.closed_at,
                "reason":       t.reason,
                "url":          t.url,
            } for t in reversed(trader.closed_trades[-50:])
        ],
        "sniper": {
            "capital":   round(ss['capital'], 2),
            "pnl":       round(ss['pnl'],     2),
            "n_open":    ss['n_open'],
            "n_closed":  ss['n_closed'],
            "n_won":     ss['n_won'],
            "win_rate":  round(ss['win_rate'], 3),
            "open_trades": [
                {
                    "slug":        t.slug,
                    "direction":   t.direction,
                    "entry_price": round(t.entry_price, 4),
                    "vol_ratio":   round(t.vol_ratio,   1),
                    "move":        round(t.move,        3),
                    "entered_at":  t.entered_at,
                    "end_ts":      t.end_ts,
                    "status":      "open",
                    "profit":      0,
                    "roi_pct":     0,
                } for t in sniper.open_trades.values()
            ],
            "closed_trades": [
                {
                    "slug":        t.slug,
                    "direction":   t.direction,
                    "entry_price": round(t.entry_price, 4),
                    "vol_ratio":   round(t.vol_ratio,   1),
                    "move":        round(t.move,        3),
                    "entered_at":  t.entered_at,
                    "end_ts":      t.end_ts,
                    "outcome":     t.outcome,
                    "status":      t.status,
                    "profit":      round(t.profit,  2),
                    "roi_pct":     round(t.roi_pct, 1),
                } for t in sniper.closed_trades[-30:]
            ],
        },
        "alerts":         alerts[-30:],
        "collector_rows": n_rows,
    }

    tmp = DASHBOARD_JSON + ".tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, DASHBOARD_JSON)   # atomic write — no half-reads
    except Exception as e:
        logging.getLogger("dashboard").error(f"JSON write: {e}")


def div(n=None): return gray("─"*(n or W-4))

def _col_widths():
    # Market, Side, Entry, Now, In$, P&L, ROI, TimeLeft
    return 30, 6, 8, 8, 9, 10, 7, 10

def render(trader:CopyTrader,sniper:SniperTester,start_time:float,alerts:List[str]):
    s   = trader.summary()
    ss  = sniper.summary()
    ps  = sorted(trader.positions.values(),key=lambda p:p.entry_amount,reverse=True)
    cts = trader.closed_trades[-6:]
    up  = time.time()-start_time
    h=int(up//3600); m=int((up%3600)//60); sc=int(up%60)

    pct    = s['invested']/s['budget']*100 if s['budget'] else 0
    bar_w  = 24; filled=int(pct/100*bar_w)
    bar    = C.YL+"█"*filled+C.GY+"░"*(bar_w-filled)+C.R
    spnl   = s['available']-s['session_start']
    n_rows = sum(1 for _ in open(COLLECTOR_FILE))-1 if os.path.exists(COLLECTOR_FILE) else 0

    bgt=bold(f"${s['budget']:.2f}"); avl=green(f"${s['available']:.2f}")
    inv=yel(f"${s['invested']:.2f}"); ret=cyan(f"${s['returned']:.2f}")
    rlz=pnlc(s['realized'],f"${s['realized']:+.2f}")
    unr=pnlc(s['unrealized'],f"${s['unrealized']:+.2f}")
    sp=pnlc(spnl,f"${spnl:+.2f}")

    CM,CS,CE,CN,CI,CP,CR,CT=_col_widths()
    L=[]

    # ── Header ────────────────────────────────────────────────────────────────
    L+=[f"{C.CY}╔{'═'*(W-2)}╗{C.R}",
        (f"{C.CY}║{C.R}  {bold('POLYMARKET COPY TRADER  ─  PAPER MODE'):<46}"
         f"  uptime {gray(f'{h:02d}:{m:02d}:{sc:02d}')}"
         f"  {yel(TARGET_WALLET[:20]+'...')}"
         f"  {C.CY}║{C.R}"),
        f"{C.CY}╚{'═'*(W-2)}╝{C.R}",""]

    # ── Copy Trader Budget ────────────────────────────────────────────────────
    L+=[f"  {green('▼ COPY TRADER BUDGET')}",f"  {div(90)}",
        f"  Portfolio {bgt}   Available {avl}   Invested {inv} ({pct:.0f}%)  {bar}",
        f"  Realized  {rlz}   Unrealized {unr}   Returned {ret}   Session {sp}",""]

    # ── Open Positions ────────────────────────────────────────────────────────
    hint=gray("type [number]+Enter to close")
    L+=[f"  {C.YL}▼ OPEN POSITIONS  ({s['n_open']})  {hint}{C.R}",
        f"  {div()}",
        # Header row — fixed widths, all right-aligned numbers
        f"  {bold('[#]'):4} "
        f"{bold('Market'):{CM}} "
        f"{bold('Side'):^{CS}} "
        f"{bold('Entry'):>{CE}} "
        f"{bold('Now'):>{CN}} "
        f"{bold('In$'):>{CI}} "
        f"{bold('P&L'):>{CP}} "
        f"{bold('ROI'):>{CR}} "
        f"{bold('Time Left'):>{CT}}  "
        f"{bold('URL')}",
        f"  {div()}"]

    if ps:
        for i,p in enumerate(ps[:10]):
            tl_str,_=time_left_from_ts(p.end_ts)
            roi_v=p.roi_pct; pnl_s=pnlc(p.pnl,f"${p.pnl:+.2f}")
            roi_s=pnlc(roi_v,f"{roi_v:+.1f}%")
            side_c=green(p.outcome) if p.outcome.lower() in ("yes","up") else red(p.outcome)
            num_c=cyan(f"[{i}]")
            L.append(
                f"  {pad(num_c,4,'>')} "
                f"{trunc(p.market_title,CM):{CM}} "
                f"{pad(side_c,CS,'^')} "
                f"{pad(gray(f'${p.entry_price:.4f}'),CE,'>')} "
                f"{pad(cyan(f'${p.cur_price:.4f}'),CN,'>')} "
                f"{pad(yel(f'${p.entry_amount:.2f}'),CI,'>')} "
                f"{pad(pnl_s,CP,'>')} "
                f"{pad(roi_s,CR,'>')} "
                f"{pad(tl_str,CT,'>')}  "
                f"{blue(p.url)}")
    else:
        L.append(f"  {gray('No open positions.')}")
    L.append("")

    # ── Closed Trades ─────────────────────────────────────────────────────────
    # Column widths mirror open positions exactly, replacing Time Left with Closed At
    CC = 8   # closed-at time column width
    L+=[f"  {C.GR}▼ CLOSED TRADES  ({s['n_closed']} total — last 6){C.R}",
        f"  {div()}",
        f"  {bold('[#]'):4} "
        f"{bold('Market'):{CM}} "
        f"{bold('Side'):^{CS}} "
        f"{bold('Entry'):>{CE}} "
        f"{bold('Exit'):>{CN}} "
        f"{bold('In$'):>{CI}} "
        f"{bold('Out$'):>{CP}} "
        f"{bold('Profit'):>{CR}} "
        f"{bold('ROI'):>7}  "
        f"{bold('How'):<8}  "
        f"{bold('Closed'):>{CC}}  "
        f"{bold('URL')}",
        f"  {div()}"]

    if cts:
        for i,t in enumerate(reversed(cts)):
            roi_s  = pnlc(t.roi_pct,      f"{t.roi_pct:+.1f}%")
            pnl_s  = pnlc(t.realized_pnl, f"${t.realized_pnl:+.2f}")
            side_c = green(t.outcome) if t.outcome.lower() in ("yes","up") else red(t.outcome)
            rsn_c  = (orange(t.reason) if t.reason=="manual"
                      else red(t.reason)    if t.reason=="expired"
                      else gray(t.reason))
            closed_time = gray(datetime.fromtimestamp(t.closed_at).strftime('%H:%M:%S'))
            num_c  = cyan(f"[{i}]")
            L.append(
                f"  {pad(num_c,4,'>')} "
                f"{trunc(t.market_title,CM):{CM}} "
                f"{pad(side_c,CS,'^')} "
                f"{pad(gray(f'${t.entry_price:.4f}'),CE,'>')} "
                f"{pad(cyan(f'${t.exit_price:.4f}'),CN,'>')} "
                f"{pad(yel(f'${t.entry_amount:.2f}'),CI,'>')} "
                f"{pad(cyan(f'${t.exit_amount:.2f}'),CP,'>')} "
                f"{pad(pnl_s,CR,'>')} "
                f"{pad(roi_s,7,'>')}  "
                f"{pad(rsn_c,8)}  "
                f"{pad(closed_time,CC,'>')}  "
                f"{blue(t.url)}")
    else:
        L.append(f"  {gray('No closed trades yet.')}")
    L.append("")

    # ── Sniper Tester ─────────────────────────────────────────────────────────
    sp_pnl   = ss['pnl']
    sp_wr    = ss['win_rate']
    sp_cap   = ss['capital']
    sp_sess  = sp_cap - ss['session_start']
    cap_c    = cyan(f"${sp_cap:.2f}")
    pnl_c    = pnlc(sp_pnl,f"${sp_pnl:+.2f}")
    wr_c     = green(f"{sp_wr:.0%}") if sp_wr>=0.6 else red(f"{sp_wr:.0%}") if ss['n_closed']>0 else gray("N/A")
    sess_c   = pnlc(sp_sess,f"${sp_sess:+.2f}")

    L+=[f"  {orange('▼ SNIPER TESTER  (volume-spike signal — separate $100 budget)')}",
        f"  {div(90)}",
        f"  Capital {cap_c}   Session {sess_c}   "
        f"Closed {yel(str(ss['n_closed']))}   Won {green(str(ss['n_won']))}   "
        f"Win rate {wr_c}   Total P&L {pnl_c}"]

    if sniper.open_trades:
        L.append(f"  {gray('Open sniper positions:')}")
        CS2,CE2,CN2,CT2=6,7,7,10
        L.append(
            f"    {bold('Slug'):>12}  "
            f"{bold('Dir'):^{CS2}}  "
            f"{bold('Entry'):>{CE2}}  "
            f"{bold('Vol Ratio'):>9}  "
            f"{bold('Move'):>6}  "
            f"{bold('Time Left'):>{CT2}}")
        for slug,t in sniper.open_trades.items():
            tl_s,_=time_left_from_ts(t.end_ts)
            dir_c=green(t.direction) if t.direction=="UP" else red(t.direction)
            L.append(
                f"    {slug[-12:]:>12}  "
                f"{pad(dir_c,CS2,'^')}  "
                f"{t.entry_price:>{CE2}.4f}  "
                f"{t.vol_ratio:>9.0f}x  "
                f"{t.move:>+6.2f}  "
                f"{pad(tl_s,CT2,'>')}")

    if sniper.closed_trades:
        last=sniper.closed_trades[-4:]
        L+=[f"  {gray('Last sniper trades:')}",
            f"    {bold('Time'):>8}  {bold('Slug'):>12}  {bold('Dir'):^6}  "
            f"{bold('Entry'):>7}  {bold('Result'):>7}  {bold('P&L'):>8}  {bold('Vol'):>6}x"]
        for t in reversed(last):
            dt=datetime.fromtimestamp(t.entered_at).strftime('%H:%M:%S')
            dir_c=green(t.direction) if t.direction=="UP" else red(t.direction)
            res_c=green("WIN") if t.status=="won" else red("LOSS")
            pnl_c2=pnlc(t.profit,f"${t.profit:+.2f}")
            L.append(
                f"    {dt:>8}  {t.slug[-12:]:>12}  "
                f"{pad(dir_c,6,'^')}  "
                f"{t.entry_price:>7.4f}  "
                f"{pad(res_c,7,'^')}  "
                f"{pad(pnl_c2,8,'>')}  "
                f"{t.vol_ratio:>6.0f}x")

    L+=[f"  {gray(f'Collector: {n_rows} snapshots → {COLLECTOR_FILE}  |  Trades → {SNIPER_FILE}')}",""]

    # ── Alerts ────────────────────────────────────────────────────────────────
    L+=[f"  {C.MG}▼ ALERTS{C.R}",f"  {div(80)}"]
    if alerts:
        for a in alerts[-6:]:
            c=(C.GR if "NEW BET" in a
               else C.OR if "SNIPER" in a
               else C.RE if any(x in a for x in ("CLOSED","EXPIRED","MANUAL"))
               else C.YL)
            L.append(f"  {c}{a}{C.R}")
    else:
        L.append(f"  {gray('Monitoring...')}")
    L.append("")
    L.append(f"  {gray('Refreshes every')} {yel(str(POLL_INTERVAL)+'s')} "
             f"{gray('|')} {mg('[number]+Enter = manual close')} "
             f"{gray('| Ctrl+C = close all + save')}")

    os.system('cls' if os.name=='nt' else 'clear')
    print("\n".join(L))
    sys.stdout.flush()

# ── INPUT THREAD ──────────────────────────────────────────────────────────────

def input_thread_fn(trader:CopyTrader):
    while True:
        try:
            line=input().strip()
            if line.isdigit():
                idx=int(line)
                # MUST use same sort as render: by entry_amount descending
                ps=sorted(trader.positions.values(),key=lambda p:p.entry_amount,reverse=True)
                if 0<=idx<len(ps):
                    pos=ps[idx]
                    trader.manual_queue.append(pos.key)
                    print(f"\n  {orange(f'Queued close for [{idx}] {trunc(pos.market_title,40)}')}")
                else:
                    print(f"\n  {red(f'Invalid [{idx}] — valid: 0-{len(ps)-1}')}")
        except (EOFError,KeyboardInterrupt): break
        except: pass

# ── LOGGING ───────────────────────────────────────────────────────────────────

def setup_logging():
    os.makedirs("logs",exist_ok=True)
    fh=logging.FileHandler(LOG_FILE,mode='a',encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    root=logging.getLogger(); root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        if not isinstance(h,logging.FileHandler): root.removeHandler(h)
    root.addHandler(fh)

# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main():
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11),7)
    except: pass

    setup_logging()
    logger=logging.getLogger("main")
    start_time=time.time()
    alerts:List[str]=[]

    memory=load_memory()
    trader=CopyTrader(TARGET_WALLET,memory)
    sniper=SniperTester()

    print(f"\n  {bold('POLYMARKET COPY TRADER + SNIPER TESTER')}")
    print(f"  Copy trader balance: {cyan(f'${trader.available:.2f}')}   "
          f"Sniper budget: {orange(f'${sniper.capital:.2f}')}\n")

    raw=trader._fetch_raw(); trader._prev_target=raw
    print(f"  Target: {C.WH}{len(raw)}{C.R} existing positions (not copied)")
    print(f"  {green('Watching for new bets + volume spikes...')}\n")
    await asyncio.sleep(2)

    threading.Thread(target=input_thread_fn,args=(trader,),daemon=True).start()

    n_existing=sum(1 for _ in open(COLLECTOR_FILE))-1 if os.path.exists(COLLECTOR_FILE) else 0
    print(f"  {green('Data collector running')} — {n_existing} existing snapshots in {COLLECTOR_FILE}")
    asyncio.create_task(_run_collector(sniper,logger))
    await asyncio.sleep(1)

    try:
        while True:
            events=trader.sync()
            for kind,msg in events:
                ts=datetime.now().strftime('%H:%M:%S')
                alerts.append(f"[{ts}] {msg.split(chr(10))[0]}")
                alerts=alerts[-20:]
                logger.info(msg.replace('\n',' | '))
                if kind in ("new","close"):
                    c=C.GR if kind=="new" else C.RE
                    print(f"\n{c}{'─'*65}{C.R}")
                    for line in msg.split('\n'): print(f"{c}{line}{C.R}")
                    print(f"{c}{'─'*65}{C.R}\n")
                    if kind=="new": await asyncio.sleep(3)

            # Check sniper alerts
            for slug,t in list(sniper.open_trades.items()):
                key=f"sniper_{slug}"
                if key not in [a for a in alerts]:
                    ts=datetime.now().strftime('%H:%M:%S')
                    dir_s=f"{C.GR}UP{C.R}" if t.direction=="UP" else f"{C.RE}DOWN{C.R}"
                    alerts.append(f"[{ts}] {orange('SNIPER')} {dir_s} {slug[-10:]} "
                                  f"entry={t.entry_price:.3f} vol={t.vol_ratio:.0f}x")

            save_memory(trader.to_memory())
            write_dashboard_json(trader, sniper, start_time, alerts)
            render(trader, sniper, start_time, alerts)
            await asyncio.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        write_dashboard_json(trader, sniper, start_time, alerts)
        print(f"\n{C.YL}Shutting down...{C.R}")
        closed=trader.close_all(reason="shutdown")
        for ct in closed:
            pnl_s=pnlc(ct.realized_pnl,f"${ct.realized_pnl:+.2f}")
            print(f"  Closed: {trunc(ct.market_title,45)} | {pnl_s}")

        save_memory(trader.to_memory())
        s=trader.summary(); ss=sniper.summary()
        spnl=s['available']-trader.session_start
        avl_f=green(f"${s['available']:.2f}"); bgt_f=cyan(f"${s['budget']:.2f}")
        sp_f=pnlc(spnl,f"${spnl:+.2f}"); rlz_f=pnlc(s['realized'],f"${s['realized']:+.2f}")

        print(f"\n{'─'*55}")
        print(bold("FINAL REPORT"))
        print(f"{'─'*55}")
        print(f"Copy Trader:")
        print(f"  Portfolio:    {bgt_f}")
        print(f"  Session P&L:  {sp_f}")
        print(f"  All Realized: {rlz_f}")
        print(f"  Trades:       {yel(str(s['n_closed']))}")
        sn_cap  = cyan(f"${ss['capital']:.2f}")
        sn_pnl  = pnlc(ss['pnl'], f"${ss['pnl']:+.2f}")
        sn_wr   = f"{ss['win_rate']:.0%}"
        print(f"Sniper Tester:")
        print(f"  Capital:      {sn_cap}")
        print(f"  Trades:       {ss['n_closed']}  Win rate: {sn_wr}")
        print(f"  P&L:          {sn_pnl}")
        print(f"\n{green('Memory saved. Next run continues from here.')}")

if __name__=="__main__":
    asyncio.run(main())