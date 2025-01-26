select
   asofdate,
   z.symbol,
   x.description, 
   notional,
   last_px,
   loadtime,
   folioname,
   username   
from 
   v_stage_trade_pos_1 z 
   left join master_security x on 
      z.symbol = x.symbol
where
   z.folioname =  %(folioname)s and 
   z.username = %(username)s
order by 
   z.symbol       

