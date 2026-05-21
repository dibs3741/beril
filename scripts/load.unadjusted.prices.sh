curl -k -X PUT "https://localhost:443/folio/prices/v2" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "'"$1"'"}'

