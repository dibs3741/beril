delete from stage_trade_pos_1 where folioname = 'ira' and username = 'sample@berilsoft.com'; 
insert into stage_trade_pos_1( 
   asofdate,
   loaddate,
   symbol,
   notional,
   last_px,
   batchid,
   folioname,
   username)
select 
   asofdate,
   loaddate,
   symbol,
   notional,
   last_px,
   batchid,
   folioname,
   'sample@berilsoft.com' username
from 
   stage_trade_pos_1
where    
   folioname = 'ira' and 
   username = 'dibyendu@gmx.com' 
   
