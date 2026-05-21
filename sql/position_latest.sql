select
   z.asofdate,
   z.symbol,
   x.description, 
   notional,
   COALESCE(p.price, 0) last_px,
   loadtime,
   folioname,
   username   
from 
   v_stage_trade_pos_1 z 
   left join master_security x on 
      z.symbol = x.symbol
   left join v_prices_unadjusted p on 
      z.symbol = p.ticker
where
   z.folioname =  %(folioname)s and 
   z.username = %(username)s
order by 
   z.symbol       

