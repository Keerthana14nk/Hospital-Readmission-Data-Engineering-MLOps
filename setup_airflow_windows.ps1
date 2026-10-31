# setup_airflow_windows.ps1
# Run ONCE to initialize Airflow on Windows.
# Then use start_airflow_windows.ps1 to launch the web UI.

$PROJECT = Split-Path $MyInvocation.MyCommand.Path
$AIRFLOW_HOME = "$PROJECT\airflow_windows"
$AIRFLOW_DAGS = "$PROJECT\airflow\dags"

Write-Host "Setting up Airflow for Windows..." -ForegroundColor Cyan
Write-Host "AIRFLOW_HOME = $AIRFLOW_HOME" -ForegroundColor Yellow

# Set env vars
$env:AIRFLOW_HOME = $AIRFLOW_HOME
$env:PYTHONPATH   = $PROJECT

# Create airflow_windows home directory
if (-Not (Test-Path $AIRFLOW_HOME)) {
    New-Item -ItemType Directory -Path $AIRFLOW_HOME | Out-Null
}

# Initialize the DB
Write-Host "Initializing Airflow database..." -ForegroundColor Yellow
airflow db migrate

# Create admin user
Write-Host "Creating admin user (username: admin, password: admin)..." -ForegroundColor Yellow
airflow users create `
    --username admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com `
    --password admin

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Now run:  .\start_airflow_windows.ps1" -ForegroundColor Green
Write-Host ""
