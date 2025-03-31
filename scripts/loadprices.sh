echo current time is: `date +%Y-%m-%d\ %H:%M:%S`
echo loading prices for $1....
/usr/bin/curl -X PUT -H "Content-Type: application/json" -d '{"symbol":"'"$1"'"}' http://localhost:8000/folio/prices/v1
echo - 
