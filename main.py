# main.py

import traceback
import logging
import numpy as np 
import pandas as pd
import datetime as dt
import numpy_financial as npf
from decimal import Decimal 
from tabulate import tabulate
from scipy.stats import norm 
from typing import Dict, List, Optional
#
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2, OAuth2PasswordRequestForm
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.handlers.sha2_crypt import sha512_crypt as crypto
from passlib.context import CryptContext
from pydantic import BaseModel
from rich import inspect, print
from rich.console import Console
from datetime import datetime, timedelta
from pytz import timezone
#
from sqlalchemy.orm import Session
#
import yfinance as yf 
import crud, models, schemas
from models import Position
from models import Trades
from models import Prices
from database import SessionLocal, engine
from auth_bearer import JWTBearer
from auth_bearer import decodeJWT

logging.getLogger('passlib').setLevel(logging.ERROR)
console = Console()
password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# --------------------------------------------------------------------------
# Models and Data
# --------------------------------------------------------------------------
class User(BaseModel):
    username: str
    hashed_password: str

# Create a "database" to hold your data. This is just for example purposes. In
# a real world scenario you would likely connect to a SQL or NoSQL database.
class DataBase(BaseModel):
    user: List[User]

DB = DataBase(
    user=[
        User(username="user1@gmail.com", hashed_password=crypto.hash("12345")),
        User(username="user2@gmail.com", hashed_password=crypto.hash("12345")),
    ]
)

def get_user(username: str) -> User:
    lines = [] 
    users = [] 
    with open('./db/users.csv') as file:
        lines = [line.rstrip() for line in file]
    for line in lines: 
        user, email = line.split(',', 1) 
        users.append(User(username=user, hashed_password=email)) 
    user = [user for user in users if user.username == username]
    #user = [user for user in DB.user if user.username == username]
    if user:
        return user[0]
    return None

# --------------------------------------------------------------------------
# Setup FastAPI
# --------------------------------------------------------------------------
class Settings:
    SECRET_KEY: str = "secret-key"
    REFRESH_SECRET_KEY: str = "secret-key"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30  # in mins
    REFRESH_TOKEN_EXPIRE_MINUTES = 30  # in mins
    COOKIE_NAME = "access_token"

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name='static')
settings = Settings()

# --------------------------------------------------------------------------
# Authentication logic
# --------------------------------------------------------------------------

def create_access_token(data: Dict) -> str:
    to_encode = data.copy()
    expire = dt.datetime.utcnow() + dt.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.SECRET_KEY, 
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def authenticate_user(username: str, plain_password: str) -> User:
    user = get_user(username)
    if not user:
        return False
    if not crypto.verify(plain_password, user.hashed_password):
        return False
    return user

def remove_prefix(input_string, suffix):
    if suffix and input_string.startswith(suffix):
        return input_string[len(suffix):]
    return input_string

def decode_token(token: str) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Could not validate credentials."
    )
    try:
        #token = token.removeprefix("Bearer").strip()
        token = remove_prefix(token, "Bearer").strip() 
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("username")
        if username is None:
            console.log(">>>>> raising credentials_exception (suppressed)")
            #raise credentials_exception
            return None 
    except JWTError as e:
        console.log(e)
        return None 
        #raise credentials_exception
    except Exception as e:
        console.log(e)
    
    user = get_user(username)
    return user


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------------------------------
# Home Page
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    console.log(">>>>> from inside home")
    context = {
        "user": None,
        "request": request,
    }
    return templates.TemplateResponse("index.html", context)


# --------------------------------------------------------------------------
# Private Page
# --------------------------------------------------------------------------
# A private page that only logged in users can access.

@app.get("/private", response_class=HTMLResponse)
#def private_get(request: Request, dependencies=Depends(JWTBearer())): 
def private_get(request: Request):
    return templates.TemplateResponse("private1.html", {"request":request, "user":None})

#@app.get("/private", response_class=HTMLResponse)
#def private_get(request: Request, user: User = Depends(get_current_user_from_token)):
#    console.log(">>>>> from inside private")
#    context = {
#        "user": user,
#        "request": request
#    }
#    if user is None:
#        return templates.TemplateResponse("login.html", context)
#    try:
#        console.log(">>>>> from inside private, user login cache found")
#        return templates.TemplateResponse("private1.html", context)
#    except Exception as e:
#        console.log(">>>>> from inside private - exception")
#        console.log(traceback.format_exc())
#        console.log(e)
#        return templates.TemplateResponse("login.html", context)

# --------------------------------------------------------------------------
# Login - GET
# --------------------------------------------------------------------------
@app.get("//login", response_class=HTMLResponse)
def login_get(request: Request):
    console.log(">>>>> from inside auth/login(get)")
    context = {
        "request": request,
    }
    return templates.TemplateResponse("login.html", context)


# --------------------------------------------------------------------------
# Login - POST
# --------------------------------------------------------------------------

@app.post("/auth/login/v2") 
async def auth_login_v2(request: schemas.requestdetails, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(email=request.email).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Incorrect email"
        )
    hashed_pass = user.password
    if not password_context.verify(request.password, hashed_pass): 
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )
    
    expires_delta = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expires_delta, "sub": str(user.username)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, settings.ALGORITHM)

    token_db = models.TokenTable(
        user_id       = user.id,  
        username      = user.username,  
        access_token  = encoded_jwt,  
        refresh_token = encoded_jwt, 
        status        = True)
    db.add(token_db)
    db.commit()
    db.refresh(token_db)

    return {
        "access_token": encoded_jwt,
        "refresh_token": encoded_jwt,
    }

@app.get("/auth/register", response_class=HTMLResponse)
def register_get(request: Request):
    context = {
        "request": request,
    }
    return templates.TemplateResponse("register.html", context)

@app.post("/auth/register", response_class=HTMLResponse)
async def register_post(request: Request):
    form = LoginForm(request)
    await form.load_data()
    if await form.is_valid():
        try:
            hashpwd = crypto.hash(form.password) 
            df = pd.DataFrame.from_dict({'username':[form.username], 'userpwd':[hashpwd]}) 
            df.to_csv('./db/users.csv', encoding='utf-8', index=False)
            #response = RedirectResponse("/auth/login", status.HTTP_302_FOUND)
        except HTTPException:
            form.__dict__.update(msg="")
            form.__dict__.get("errors").append("Incorrect Email or Password")
            return templates.TemplateResponse("register.html", form.__dict__)
    return templates.TemplateResponse("register.html", form.__dict__)

@app.post("/auth/register/v2")
def register_post_v2(user: schemas.UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter_by(email=user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, msg="Email already registered")

    encrypted_password = password_context.hash(user.password)
    new_user = models.User(username=user.email, email=user.email, password=encrypted_password )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {'data':'Registration Successful! Now Login'} 

# --------------------------------------------------------------------------
# Logout
# --------------------------------------------------------------------------
@app.get("/auth/logout", response_class=HTMLResponse)
def login_get():
    response = RedirectResponse(url="/")
    response.delete_cookie(settings.COOKIE_NAME)
    return response

#@app.get("/users/", response_model=List[schemas.User])
@app.get("/users/") 
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return {'data':users} 

@app.get("/folios/v1", response_model=List[schemas.MasterFolio]) 
async def folio_get_v1(db: Session = Depends(get_db), dependencies=Depends(JWTBearer())):
    u = decodeJWT(dependencies).get('sub') 
    folios = db.query(models.MasterFolio).filter_by(username=u).all()
    df = pd.DataFrame.from_records(folios, index='id', columns=['user','folio','id']) 
    console.log(decodeJWT(dependencies)) 
    return folios

@app.get("/folio/attribution/v1") 
async def folio_attribution_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub') 
    datefrom = '2024-09-30' 
    dateto = '2024-12-31' 
    file = 'sql/prices_security_alt.sql' 
    with open(file, 'r') as file:
        f = file.read() 
        params = {'datefrom':datefrom, 'dateto':dateto, 'folioname':folioname, 'username':u}
        df = pd.read_sql(f, engine, params=params)
        df21 = df[['ticker','allocated']]
        df22 = df21.drop_duplicates() 
        df23 = df22.squeeze(axis=0)
        df23.set_index("ticker", inplace = True)
        #console.log(df23) 
        #
        df2 = df[['asofdate', 'ticker', 'price']] 
        df3 = df2.pivot(index='asofdate', columns='ticker', values='price') 
        df4 = pd.DataFrame(df3.to_records())
        df4.set_index("asofdate", inplace = True)
        df5 = df4.pct_change()
        df6 = df5.round(6)[1:] 
        df6.reset_index(inplace=True)
        df6['asofdate'] = pd.to_datetime(df6['asofdate'])
        df6['yymm'] = df6['asofdate'].dt.strftime('%Y%m')
        df6.drop('asofdate', axis=1, inplace=True)
        df6.set_index("yymm", inplace = True)
        df7 = df6+1
        df8 = df7.groupby(['yymm']).cumprod()-1
        df9 = df8.groupby(['yymm']).tail(1)
        #console.log(df9) 
        #
        df30 = df9.mul(df23['allocated'].to_list(), axis=1) 
        df31 = df30.round(6) 
        df31.reset_index(inplace=True)
        df31.to_csv('./df31.csv') 
        df32 = pd.melt(df31, id_vars=['yymm'], var_name='symbol', value_name='returns') 
        df33 = df32.pivot(index='symbol', columns='yymm', values='returns') 
        df34 = pd.DataFrame(df33.to_records())
        console.log(df34) 
    return {'data':df34.to_dict(orient='records')}

@app.put("/folio/prices/v1") 
async def folio_prices_v1(payload: Request, db: Session = Depends(get_db)):
    payloadjs = await payload.json()
    asofdate = datetime.now(timezone('US/Eastern')).strftime('%Y-%m-%d')
    symbol = payloadjs["symbol"]
    #
    data = yf.download(symbol, '2022-12-30', asofdate) 
    data.reset_index(inplace=True)
    df1 = data[['Date', 'Close']]
    df1 = df1.round({'Close':2}) 
    df1.reset_index(drop=True, inplace=True)
    #
    try: 
        r = db.query(Prices).filter(
            Prices.ticker == symbol,
            ).delete() 
        db.commit()
        console.log(f"Deleted {r} records from table") 
        console.log(f"Loading {len(df1)} rows...") 
        for i, row in df1.iterrows(): 
            o = Prices()
            o.asofdate = row['Date'].item()  
            o.price = row['Close'].item()  
            o.ticker = symbol
            db.add(o) 
        db.commit()
        console.log(f"--end--") 
        console.log(f"") 
    except: 
        db.rollback()
        console.log(traceback.format_exc())

@app.put("/folio/position/v1") 
async def folio_position_v1(payload: Request, db: Session = Depends(get_db)):
    payloadjs = await payload.json()
    asofdate = datetime.now(timezone('US/Eastern')).strftime('%Y-%m-%d')
    batchid = datetime.now(timezone('US/Eastern')).strftime('%y%m%d%H%M%S')
    df10 = pd.DataFrame(payloadjs['positions'])
    df10.columns = df10.iloc[0]
    df10 = df10[1:] 
    df10.fillna(0, inplace=True)
    #
    try: 
        r = db.query(Position).filter(
            Position.asofdate == asofdate,
            Position.folioname == payloadjs['folioname'],
            Position.username == 'dibyendu@gmx.com',
            ).delete() 
        db.commit()
        console.log(f"Deleted {r} records from table") 
        console.log(f"Portfolio: {payloadjs['folioname']}") 
        console.log(f"Batch id: {batchid}") 
        console.log(f"Loading {len(df10)} rows...") 
        for i, row in df10.iterrows(): 
            o = Position()
            o.asofdate = asofdate
            o.loaddate = asofdate
            o.symbol = row['Symbol']
            o.notional = row['Notional'] 
            o.last_px = row['Last Price'] 
            o.batchid = batchid
            o.folioname = payloadjs['folioname'] 
            o.username = 'dibyendu@gmx.com' 
            db.add(o) 
            db.commit()
        console.log(f"--end--") 
        console.log(f"") 
    except: 
        db.rollback()
        console.log(traceback.format_exc())


@app.put("/folio/trades/v1") 
async def folio_trades_v1(payload: Request, db: Session = Depends(get_db)):
    payloadjs = await payload.json()
    asofdate = datetime.now(timezone('US/Eastern')).strftime('%Y-%m-%d')
    batchid = datetime.now(timezone('US/Eastern')).strftime('%y%m%d%H%M%S')
    df10 = pd.DataFrame(payloadjs['positions'])
    df10.columns = df10.iloc[0]
    df10 = df10[1:] 
    df10.dropna(how='all', inplace=True)
    console.log(df10.to_string()) 
    df10.fillna(0, inplace=True)
    #
    try: 
        for i, row in df10.iterrows(): 
            o = Trades()
            o.loaddate = asofdate
            o.batchid = batchid
            o.user_name = payloadjs['username']
            o.asofdate = row['Run Date'] 
            o.account = row['Account'] 
            o.symbol = row['Symbol']
            o.quantity = row['Quantity']
            db.add(o) 
        db.commit()
    except: 
        db.rollback()
        console.log(traceback.format_exc())

@app.get("/folio/projection/v1") 
def folio_projection_v1(
        folioname:str, 
        hold: float, 
        intrate: float, 
        mlydep: float, 
        db: Session = Depends(get_db), 
        tkn=Depends(JWTBearer())):
    #fv +
    #pv*(1+rate)**nper +
    #pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0
    #numpy_financial.fv(rate, nper, pmt, pv, when='end')
    #10 years of saving
    #starting with 100$ in account 
    #additional monthly savings of $100
    #interest rate is 5% (annually) compounded monthly
    #p = npf.fv(0.05/12, 10*12, -100, -100)
    #
    console.log(f'holding period: {hold}') 
    console.log(f'interest rate(yearly): {intrate}') 
    console.log(f'monthly deposit: {mlydep}') 
    #
    y = 0 
    u = decodeJWT(tkn).get('sub') 
    df = pd.DataFrame()
    file = 'sql/cashholdings.sql'
    with open(file, 'r') as file:
        f = file.read()
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u})
        y = df.notional.sum()
    console.log(f'portfolio nav: {y}') 

    p = npf.fv(intrate/100/12, hold*12, -1*mlydep, -1*y)
    console.log(f'projected portfolio nav: {p}') 
    return {'data':p} 

@app.get("/folio/tradereq/v1") 
def folio_tradereq_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub')
    spreadlimitup = 3000
    spreadlimitdn = -3000
    tradinglimit = 1000
    df1 = pd.DataFrame()
    r = folio_dashboard_v1(folioname, db, tkn) 
    df1 = pd.DataFrame.from_dict(r['data'], orient='columns') 
    df2 = df1[~df1.sector.isin(['cash'])]
    df3 = df2[~((df2.spread > 0) & (df2.spread < spreadlimitup))]
    df4 = df3[~((df3.spread < 0) & (df3.spread > spreadlimitdn))]
    #
    x = position_latest_v1(folioname, db, tkn) 
    df10 = pd.DataFrame.from_dict(x['data'], orient='columns') 
    df11 = df10[['symbol', 'last_px']]
    #
    # merge the two frames to bring price and spread together 
    df4.set_index("symbol", inplace = True)
    df11.set_index("symbol", inplace = True)
    df20 = pd.merge(df4, df11, on=['symbol'], how='inner')
    #
    # introduce a side column 
    df20['side'] = 'buy' 
    df20.loc[df20['spread'] < 0, 'side'] = 'sell'
    #
    # calculate qty to trade 
    df20['qty'] = 1000/df20['last_px'] 
    df20.qty = df20.qty.astype(int) 
    #
    df30 = pd.DataFrame()
    file = 'sql/trades.sql' 
    with open(file, 'r') as file:
        f = file.read() 
        df30 = pd.read_sql(f, engine, params={'username':u}) 
        df30.set_index("symbol", inplace = True)
    #
    df40 = pd.merge(df20, df30, on=['symbol'], how='left')
    df40.reset_index(inplace=True)
    df40.fillna('', inplace=True)
    df41 = df40[['symbol', 'side', 'qty', 'asofdate', 'quantity', 'account']] 
    console.log(df41.to_string())
    return {'data':df41.to_dict(orient='records')}

@app.get("/folio/cash/v1") 
def folio_cash_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub') 
    df = pd.DataFrame()
    file = 'sql/cashholdings.sql' 
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
    x = df[df['holdings'] == 'cash'].notional.item()
    y = df.notional.sum()
    return {'data':x/y} 

@app.get("/folio/allocation/v1") 
def folio_allocation_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub') 
    df = pd.DataFrame()
    file = 'sql/allocation.sql' 
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
    return {'data':df.to_dict(orient='records')}

@app.get("/folio/dashboard/v1") 
def folio_dashboard_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub') 
    df = pd.DataFrame()
    file = 'sql/holdings.sql' 
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        df.rename(columns={'notional':'actual'}, inplace=True)
    file2 = 'sql/investablenav.sql' 
    investablenav = 0 
    with open(file2, 'r') as file:
        f = file.read() 
        df1 = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        investablenav = df1['notional'][0] 
    console.log(f'>>>> {investablenav}') 
    df['forecast'] = df['allocated'] * investablenav * 0.01
    df['spread'] = df['forecast'] - df['actual'] 
    df.loc[df['sector'] == 'cash', 'spread'] = 0 
    console.log(f'>>>> {df}') 
    return {'data':df.to_dict(orient='records')}

@app.get("/position/latest/v1") 
def position_latest_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub') 
    df = pd.DataFrame()
    file = 'sql/position_latest.sql'
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        df.fillna(0, inplace=True)
    df1 = df[['symbol','notional','last_px','description']]
    return {'data':df1.to_dict(orient='records')}

@app.get("/rebalance/v1") 
def rebalance_v1(folioname:str, db: Session = Depends(get_db)):
    bar = 2000
    df = pd.DataFrame()
    r1 = folio_dashboard_v1(folioname) 
    df10 = pd.DataFrame.from_dict(r1['data'])
    console.log(f'\n{df10}') 
    df11 = df10[~df10.symbol.isin(['CORE**','SHV'])]
    df12 = df11[abs(df11.spread) > bar]
    df12['trade'] = df12['spread'].apply(lambda x:x-bar if x>0 else x+bar)  

    r2 = position_latest_v1() 
    df20 = pd.DataFrame.from_dict(r2['data'])
    df21 = df20[['symbol','last_px']]
    #
    df30 = pd.merge(df12, df21, on='symbol', how='inner') 
    df30['lots'] = round(df30['trade']/df30['last_px']) 
    df31 = df30[['symbol', 'lots']] 
    df32 = df31[abs(df31.lots) > 0]
    #
    return {'data':df32.to_dict(orient='records')}

@app.get("/folio/drawdowns/v1") 
def drawdowns_v1(folioname:str, db: Session = Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub')
    file = 'sql/activity.sql'
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        df.fillna(0, inplace=True)
    #
    activity1 = df[['asofdate', 'notional_start', 'income', 'pnl']] 
    activity2 = activity1[1:].copy() 
    activity2.loc[:, 'netpnl'] = activity2['income'] + activity2['pnl']
    activity2.loc[:, 'return'] = activity2['netpnl'] / activity2['notional_start']
    activity3 = activity2[['asofdate', 'return']] 
    #
    #- drawdowns
    activity3.set_index('asofdate', inplace=True) 
    df30 = (activity3+1).cumprod() * 100 
    activity3.reset_index(inplace=True)
    #
    drawdowns = df30/df30.expanding(min_periods=1).max()-1
    is_zero = drawdowns == 0
    #- find start dates (first day where dd is non-zero after a zero)
    start = ~is_zero & is_zero.shift(1)
    start.iloc[0] = False
    start = start.index[start.iloc[:,0]].tolist()
    #- find end dates (first day where dd is 0 after non-zero)
    end = is_zero & (~is_zero).shift(1)
    end.iloc[0] = False
    end = end.index[end.iloc[:,0]].tolist()

    if len(start) == 0:
        return None

    # drawdown has no end (end period in dd)
    if len(end) == 0:
        end.append(drawdowns.index[-1])

    # if the first drawdown start is larger than the first drawdown end it
    # means the drawdown series begins in a drawdown and therefore we must add
    # the first index to the start series
    if start[0] > end[0]:
        start.insert(0, drawdowns.index[0])

    # if the last start is greater than the end then we must add the last index
    # to the end series since the drawdown series must finish with a drawdown
    if start[-1] > end[-1]:
        end.append(drawdowns.index[-1])

    result = pd.DataFrame(
        columns=(
            'start', 
            'end', 
            'length', 
            'drawdown', 
            'trough', 
            'daysToTrough', 
            'daysInRecovery', 
            'pctRecovery', 
            'pctRecoveryRemaining'),
        index=range(0, len(start))
    )
    for i in range(0, len(start)):
        dd = drawdowns[start[i]:end[i]].min()
        ddDateDf = drawdowns[start[i]:end[i]].idxmin()
        ddDate = ddDateDf.iloc[0]
        daysToTrough = (ddDate - start[i]).days
        daysInRecovery = (end[i] - ddDate).days
        pctRecovery = np.subtract(drawdowns.loc[end[i]], dd)
        pctRecRemain = -1 * drawdowns.loc[end[i]] / (drawdowns.loc[end[i]] + 1)
        result.iloc[i] = (
            start[i].strftime("%m-%d-%Y"), 
            end[i].strftime("%m-%d-%Y"), 
            (end[i] - start[i]).days, 
            dd.iloc[0] * 100, 
            ddDate.strftime("%m-%d-%Y"), 
            daysToTrough, 
            daysInRecovery, 
            pctRecovery.iloc[0] * 100, 
            pctRecRemain.iloc[0] * 100)
    console.log(f'\n{result}') 
    #
    return {'data':result.to_dict(orient='records')}
 
@app.get("/folio/income/mtd/v1") 
def income_mtd_v1(folioname:str, db: Session=Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub')
    file = 'sql/activity.sql'
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        df.fillna(0, inplace=True)
    #
    activity1 = df[['asofdate', 'notional_start', 'flow', 'income', 'pnl']] 
    activity2 = activity1[1:].copy() 
    activity2.loc[:, 'netpnl'] = activity2['income'] + activity2['pnl']
    activity2.loc[:, 'return'] = activity2['netpnl'] / activity2['notional_start'] * 100
    activity3 = activity2[['asofdate', 'income', 'return', 'flow']] 
    #activity3.set_index('asofdate', inplace=True) 
    #
    return {'data':activity3.tail(15).to_dict(orient='records')}

@app.get("/folio/income/ytd/v1") 
def income_ytd_v1(folioname:str, db: Session=Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub')
    file = 'sql/activity.sql'
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        df.fillna(0, inplace=True)
    #
    activity1 = df[['asofdate', 'notional_start', 'flow', 'income', 'pnl']] 
    activity2 = activity1[1:].copy() 
    activity2['year'] = pd.DatetimeIndex(activity2['asofdate']).year 
    activity2.loc[:, 'netpnl'] = activity2['income'] + activity2['pnl']
    activity2.loc[:, 'return'] = activity2['netpnl'] / activity2['notional_start'] 
    activity2['return1'] = activity2['return'] + 1 
    activity3 = activity2[['year', 'return1']] 
    activity3['ytd'] = activity3.groupby(['year'])['return1'].cumprod()-1
    activity4 = activity3.groupby(['year']).tail(1) 
    activity4.ytd = activity4.ytd * 100 
    activity5 = activity4[['year','ytd']] 
    activity5.set_index('year', inplace=True) 
    #
    activity6 = activity2[['year', 'income', 'flow']] 
    activity7 = activity6.groupby(['year']).sum() 
    activity8 = activity7.groupby(['year']).tail(1) 
    #
    activity10 = pd.merge(activity5, activity8, on=['year'], how='inner')
    activity10.reset_index(inplace=True)
    console.log(f'\n{activity5}') 
    console.log(f'\n{activity8}') 
    console.log(f'\n{activity10}') 
    #
    return {'data':activity10.to_dict(orient='records')}


@app.get("/metrics/v2") 
def metrics_v2(folioname:str, db: Session=Depends(get_db), tkn=Depends(JWTBearer())):
    u = decodeJWT(tkn).get('sub')
    file = 'sql/activity.sql'
    with open(file, 'r') as file:
        f = file.read() 
        df = pd.read_sql(f, engine, params={'folioname':folioname, 'username':u}) 
        df.fillna(0, inplace=True)
    file1 = 'sql/prices_security.sql'
    with open(file1, 'r') as file1:
        f1 = file1.read() 
        spxprices1 = pd.read_sql(f1, engine, params={'ticker':'SPX'}) 
        spxprices1.fillna(0, inplace=True)
    #
    spxprices2 = spxprices1[spxprices1['asofdate'].isin(df.asofdate.tolist())].copy() 
    spxprices2.loc[:,'return'] = spxprices2['price'].pct_change()
    spxprices3 = spxprices2[['asofdate','return']] 
    spxprices4 = spxprices3[1:] 
    spxprices4.to_csv('./spxprices4.csv') 
    spxprices4.set_index('asofdate', inplace=True) 
    #
    activity1 = df[['asofdate', 'notional_start', 'income', 'pnl']] 
    activity2 = activity1[1:].copy() 
    activity2.loc[:, 'netpnl'] = activity2['income'] + activity2['pnl']
    activity2.loc[:, 'return'] = activity2['netpnl'] / activity2['notional_start']
    activity3 = activity2[['asofdate', 'return']] 
    activity3.set_index('asofdate', inplace=True) 
    #
    #- volatility 
    #- correlation
    #- tracking error 
    #- avg ret of pos mths
    #- avg ret of neg mths
    stratvol = activity3.std(axis=0) * np.sqrt(12)
    corr = activity3.corrwith(spxprices4.iloc[:,0])
    trackerr = (np.subtract(activity3, spxprices4)).std(axis=0) 
    avgofposmonth = activity3[activity3['return'] > 0].mean() 
    avgofnegmonth = activity3[activity3['return'] < 0].mean() 
    #
    #- cagr, max drawdown 
    activity3.reset_index(inplace=True)
    yearfrac = ((activity3.index[-1] - activity3.index[0])+2)/12 
    activity3.set_index('asofdate', inplace=True) 
    df30 = (activity3+1).cumprod() * 100 
    cagrstrat = ((df30.iloc[-1]/100) ** (1 / yearfrac)) -1 
    maxdrawdown = (df30/df30.expanding(min_periods=1).max()).min()-1 
    #
    #- drawdown details 
    #- another way, for record 
    #roll_max = np.maximum.accumulate(df30)
    #drawdowns_alt = df30 / roll_max - 1.
    #console.log(f'drawdowns(alt): \n{drawdowns_alt}') 
    #
    #perposmonth = df2[df2['strat']>0].count().item()/df2.count().item()
    #
    #- sharpe ratio
    #- sortino ratio
    #- var95
    df22 = activity3.copy() 
    df22.loc[:, 'rfr'] = 0.02
    df22.loc[:, 'xcessret'] = df22['return'] - df22['rfr'] 
    df23 = df22[['xcessret']] 
    sharperatio = df23.mean().item()/df23.std().item() 
    sortinoratio = df23.mean().item()/df23[df23.xcessret < 0].std().item() 
    var95_covar = norm.ppf(0.05, activity3.mean().item(), activity3.std().item())
    var95_hist = np.percentile(activity3, 5)
    #
    console.log(f'volatility: {stratvol.item()}') 
    console.log(f'correlation: {corr.item()}') 
    console.log(f'tracking err: {trackerr.item()}') 
    console.log(f'avg ret of pos mths: {avgofposmonth.item()}') 
    console.log(f'avg ret of neg mths: {avgofnegmonth.item()}') 
    console.log(f'cagr: {cagrstrat.item()}') 
    console.log(f'max drawdown: {maxdrawdown.item()}') 
    console.log(f'sharpe ratio: {sharperatio}') 
    console.log(f'sortino ratio: {sortinoratio}') 
    console.log(f'var95-covariance: {var95_covar}') 
    console.log(f'var95-historical: {var95_hist}') 
    #
    r = {
        'volatility': stratvol.item(),
        'correlation': corr.item(),
        'trackerr': trackerr.item(),
        'avg_ret_pos_mths': avgofposmonth.item(),
        'avg_ret_neg_mths': avgofnegmonth.item(),
        'cagr': cagrstrat.item(),
        'max_drawdown': maxdrawdown.item(),
        'sharpe_ratio': sharperatio,
        'sortino_ratio': sortinoratio,
        'var95_covariance': var95_covar,
        'var95_historical': var95_hist
    }
    #
    return {'data':r}

@app.get("/metrics/v1") 
def metrics_vol_v1(db: Session = Depends(get_db)):
    df1 = pd.read_csv('./db/returns.csv', header=0) 
    df1.set_index('asofdate', inplace=True) 
    #
    df2 = df1[['strat']]
    stratvol = df2.std(axis=0) * np.sqrt(12)
    #
    df3 = df1[['snp500']]
    bmarkvol = df3.std(axis=0) * np.sqrt(12)
    #
    r = df3.iloc[:,0]
    corr = df2.corrwith(r)
    #
    te1 = np.subtract(df2, df3)
    te2 = df2.std(axis=0) * np.sqrt(12)
    #
    df30 = (df2+1).cumprod() * 100 
    df30 = df30.reset_index() 
    start = df30.index[0]
    end = df30.index[-1]
    yearfrac = (end-start)/12 
    df30.set_index('asofdate', inplace=True) 
    cagrstrat = (df30.iloc[-1]/df30.iloc[0]) ** (1 / yearfrac -1) 
    df40 = (df3+1).cumprod() * 100 
    cagrbmark = (df40.iloc[-1]/df40.iloc[0]) ** (1 / yearfrac -1) 
    outperfann = np.subtract(cagrstrat.iloc[0], cagrbmark.iloc[0]) 
    infratio = np.divide(outperfann, te2) 
    #
    stratret1 = (df2+1).cumprod() - 1
    bmarkret1 = (df3+1).cumprod() - 1
    stratret2 = stratret1.iloc[-1]
    bmarkret2 = bmarkret1.iloc[-1]
    op = stratret2.item() - bmarkret2.item() 
    #
    freqmlyop1 = df2 - df3.values
    tot = len(freqmlyop1) 
    win = freqmlyop1[freqmlyop1 > 0].count() 
    freqmlyop2 = win/tot 
    #
    corroptobm = df2.corrwith(df3.iloc[:,0])
    #
    betatobm1 = np.cov(df2[df2.columns[0]], df3[df3.columns[0]])
    betatobm2 = betatobm1[0,1]/betatobm1[1,1]
    #
    avgposmonth = df2[df2['strat']>0].mean() 
    avgnegmonth = df2[df2['strat']<0].mean() 
    #
    rfr = 0 
    sharperatio = (cagrstrat.iloc[0] - rfr)/stratvol.item() 
    sortinoratio = (df2.mean().item() - rfr)/stratvol.item() 
    #
    maxdrawdown = (df30/df30.expanding(min_periods=1).max()).min()-1 
    #
    var95 = np.percentile(df2, 5)
    perposmonth = df2[df2['strat']>0].count().item()/df2.count().item()

    return {"data": {
        'stratvol' :stratvol.item(),
        'bmarkvol' :bmarkvol.item(),
        'correl'   :corr.item(),
        'trackerr' :te2.item(),
        'infratio' :infratio.item(),
        'stratret' :stratret2.item(), 
        'bmarkret' :bmarkret2.item(), 
        'outperf' : op,
        'opannual': outperfann,
        'freqmlyop': freqmlyop2.item(),
        'corroptobm': corroptobm.item(),
        'betatobm' :betatobm2,
        'stdeviation' :stratvol.item(),
        'sharperatio' :sharperatio,
        'sortinoratio' :sortinoratio,
        'avgposmonth' :avgposmonth.item(),
        'avgnegmonth' :avgnegmonth.item(),
        'maxdrawdown' :maxdrawdown.item(),
        'highwatermark' : stratret1.max()['strat'], 
        'var95' : var95,
        'perposmonth': perposmonth 
        }
    }
