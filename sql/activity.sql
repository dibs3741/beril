select
    user_name,
    folio_name,
    asofdate,
    notional_start,
    notional_end,
    flow,
    income,
    pnl
from
    activity
where
   user_name = %(username)s and 
   folio_name = %(folioname)s 
order by
   asofdate 

