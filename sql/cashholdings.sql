select 
   x.holdings, 
   sum(x.notional) notional 
from (
select 
   case 
      when f.sector = 'cash' then 'cash' else 'invested'
   end holdings, 
   v.notional
from    
   v_stage_trade_pos_1 v 
   join allocation_folio f on 
      v.symbol = f.symbol and 
      v.username = f.user_name and 
      v.folioname = f.folio_name             
where    
   v.username = %(username)s and 
   v.folioname = %(folioname)s 
)x    
group by 
   x.holdings   
