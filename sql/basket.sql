select 
   x.symbol, 
   x.sector, 
   x.allocated
from 
   allocation_folio x 
where
   x.user_name =  %(username)s and 
   x.folio_name = %(folioname)s and 
   x.allocated > 0
order by
   x.sort_order 

