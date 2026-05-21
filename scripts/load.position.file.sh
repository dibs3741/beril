echo current time is: `date +%Y-%m-%d\ %H:%M:%S`
echo loading position file....
/usr/bin/curl -k -X GET https://localhost:443/load/position/v1
echo - 
