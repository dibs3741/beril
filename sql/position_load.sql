UPDATE stage_trade_pos_1 SET last_px=?, notional=? WHERE symbol=? and asofdate=?;
INSERT INTO table (asofdate, loaddate, symbol, last_px, notional)
       SELECT 3, 'C', 'Z'
              WHERE NOT EXISTS (SELECT 1 FROM table WHERE id=3);

