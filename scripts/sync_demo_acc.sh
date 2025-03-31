export PGPASSWORD=postgres
cd /home/dmajumder/core/vayu/scripts
# /usr/bin/psql --host=localhost -U postgres -d secmaster -c 'select * from trades'
/usr/bin/psql --host=localhost -U postgres -d secmaster -a -f stage_trade_pos_1.sql 
/usr/bin/psql --host=localhost -U postgres -d secmaster -a -f stage_trade_pos_2.sql 
