exec > /home/dmajumder/core/vayu/log/fastapi.log 2>&1 
cd /home/dmajumder/core/vayu;
source env/bin/activate;
uvicorn main:app --host 0.0.0.0 --port 443 --ssl-keyfile ssl/private.key --ssl-certfile ssl/certificate.crt

