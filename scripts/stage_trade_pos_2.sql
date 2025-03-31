delete from stage_trade_pos_1 where folioname = 'account-2' and username = 'sample@berilsoft.com'; 
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
   'account-2' folioname,
   'sample@berilsoft.com' username
from 
   stage_trade_pos_1
where    
   folioname = 'brokerage' and 
   username = 'dibyendu@gmx.com' 
   
