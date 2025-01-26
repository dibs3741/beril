select
   asofdate,
   symbol,
   quantity,
   account
from 
   v_trades z 
where
   user_name = %(username)s
order by 
   symbol       
