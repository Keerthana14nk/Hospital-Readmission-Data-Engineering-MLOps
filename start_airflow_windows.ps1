# start_airflow_windows.ps1
# Starts the Airflow web server + scheduler on Windows.
# Access the UI at: http://localhost:8080
# Username: admin  |  Password: admin

$PROJECT = Split-Path $MyInvocation.MyCommand.Path
$AIRFLOW_HOME = "$PROJECT\airflow_windows"
$AIRFLOW_DAGS = "$PROJECT\airflow\dags"

$env:AIRFLOW_HOME = $AIRFLOW_HOME
$env:PYTHONPATH   = $PROJECT
$env:AIRFLOW__CORE__DAGS_FOLDER = $AIRFLOW_DAGS

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Hospital Readmission — Airflow Launcher  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "AIRFLOW_HOME : $AIRFLOW_HOME" -ForegroundColor Yellow
Write-Host "DAGs folder  : $AIRFLOW_DAGS" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting Airflow webserver + scheduler..." -ForegroundColor Yellow
Write-Host "UI will be available at: http://localhost:8080" -ForegroundColor Green
Write-Host "Username: admin   Password: admin" -ForegroundColor Green
Write-Host ""

# Launch scheduler in background
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "`$env:AIRFLOW_HOME='$AIRFLOW_HOME'; `$env:PYTHONPATH='$PROJECT'; `$env:AIRFLOW__CORE__DAGS_FOLDER='$AIRFLOW_DAGS'; airflow scheduler"
) -WindowStyle Normal

# Launch webserver in this window
airflow webserver --port 8080
