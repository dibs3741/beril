select 
   sum(notional) notional 
from 
   v_stage_trade_pos_1 
where
   username =  %(username)s and 
   folioname = %(folioname)s and 
   symbol not in ('SHV', 'SHY', 'EIPI') 

