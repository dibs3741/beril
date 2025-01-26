import datetime
from sqlalchemy import Boolean, Date, DateTime, Column, Numeric, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
#
from database import Base



class User(Base):
    __tablename__ = "auth_users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False)
    password = Column(String(100), nullable=False)

class MasterFolio(Base):
    __tablename__ = "master_folio"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False)
    folio_name = Column(String(50), nullable=False)
    
    def __iter__(self): 
        yield self.username 
        yield self.folio_name 
        yield self.id 

class TokenTable(Base):
    __tablename__ = "auth_token"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    username = Column(String(128)) 
    access_token = Column(String(256)) 
    refresh_token = Column(String(256))
    status = Column(Boolean)
    created_date = Column(DateTime, default=datetime.datetime.now)

class Position(Base):
    __tablename__ = "stage_trade_pos_1"
    id = Column(Integer, primary_key=True)
    asofdate = Column(Date) 
    loaddate = Column(Date) 
    symbol = Column(String) 
    notional = Column(Numeric(20,2)) 
    last_px = Column(Numeric(20,2)) 
    batchid = Column(Integer) 
    folioname = Column(String) 
    username = Column(String)

class Trades(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    asofdate = Column(Date) 
    loaddate = Column(Date) 
    symbol = Column(String) 
    quantity = Column(Numeric(20,2)) 
    batchid = Column(Numeric(20,2)) 
    account = Column(String) 
    user_name = Column(String)

class Prices(Base):
    __tablename__ = "prices_security"
    id = Column(Integer, primary_key=True)
    asofdate = Column(Date) 
    ticker = Column(String) 
    price = Column(Numeric(20,2)) 
