echo restarting web server....
#uvicorn main:app --reload
#uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile=./private.key --ssl-certfile=./certificate.crt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

