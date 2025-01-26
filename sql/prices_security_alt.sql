select
   s.asofdate,
   s.ticker,
   o.allocated allocated,
   s.price
from   
   prices_security s join allocation_folio o on 
      s.ticker = o.symbol 
where
   s.asofdate >= %(datefrom)s and
   s.asofdate <= %(dateto)s and 
   o.folio_name = %(folioname)s and 
   o.user_name = %(username)s 
order by
   s.asofdate,
   s.ticker

