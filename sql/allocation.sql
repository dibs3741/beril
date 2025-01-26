select
   case
      when position('usa_large' in x.sector)>0  then 'Foundational - Domestic - Large Cap'
      when position('usa_mid' in x.sector)>0  then 'Foundational - Domestic - Mid Cap'
      when position('usa_small' in x.sector)>0  then 'Foundational - Domestic - Small Cap'
      when position('developed' in x.sector)>0  then 'Foundational - Developed Markets(Non-USA)'
      when position('emerging' in x.sector)>0  then 'Foundational - Emerging Markets'
      when position('bonds' in x.sector)>0  then 'Foundational - Fixed Income, Bonds'
      when position('commodities' in x.sector)>0  then 'Specialized - Commodities, Hard Assets'
      when position('reit' in x.sector)>0  then 'Specialized - Real Estate'
      when position('alt_bdc' in x.sector)>0  then 'Alternatives - Business Development Companies(BDC)'
      when position('alt_green' in x.sector)>0  then 'Alternatives - ESG Initiatives'
      when position('alt_tech' in x.sector)>0  then 'Alternatives - Cutting Edge Technologies'
      when position('alt_utils' in x.sector)>0  then 'Alternatives - Utilities, Energy, MLP'
      else x.sector 
   end assetclass,
   x.sector sassetclass,
   x.allocated
from (
   select
      sector,
      sum(allocated) allocated 
   from 
      allocation_folio
   where 
      user_name = %(username)s and 
      folio_name = %(folioname)s 
   group by 
      sector
   )x join (
   select
      sector,
      min(sort_order) sortorder  
   from 
      allocation_folio
   where 
      user_name = %(username)s and 
      folio_name = %(folioname)s 
   group by 
      sector
   )y on 
      x.sector = y.sector    
where 
   x.sector not in ('cash') 
order by 
   y.sortorder 
            
