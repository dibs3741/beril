select
    asofdate,
    ticker,
    price
from
    prices_security
where
   ticker = %(ticker)s 
order by 
   asofdate,
   ticker
