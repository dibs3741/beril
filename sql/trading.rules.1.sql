select 
   user_name,
   folio_name,
   case
      when side = 1 then 'buy'
      when side = 2 then 'sell' 
   end side,
   trade_cap,
   drift_limit
from 
   trade_rules_1
where
   user_name = %(username)s and 
   folio_name = %(folioname)s 
