select 
   x.symbol, 
   x.sector, 
   x.allocated, 
   s.notional 
from 
   allocation_folio x join v_stage_trade_pos_1 s on 
      x.symbol = s.symbol and 
      x.folio_name = s.folioname and 
      x.user_name = s.username 
where
   x.user_name =  %(username)s and 
   x.folio_name = %(folioname)s 
order by
   x.sort_order 

