1..1000 | ForEach-Object {
    Invoke-WebRequest -Uri "http://localhost:8000" -UseBasicParsing
}
