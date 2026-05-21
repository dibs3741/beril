select
   s.asofdate,
   s.ticker,
   s.price
from   
   prices_security s 
where
   s.asofdate >= :datefrom and
   s.asofdate <= :dateto and 
   s.ticker in :tickers 
order by
   s.asofdate,
   s.ticker 

