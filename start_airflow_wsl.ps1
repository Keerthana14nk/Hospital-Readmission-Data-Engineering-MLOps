# start_airflow_wsl.ps1
# Launches Apache Airflow standalone inside WSL using the project's airflow_venv
# After running this script, open: http://localhost:8080
# Login: admin / (see airflow/standalone_admin_password.txt)

$PROJECT = "/mnt/c/Users/vke23/Downloads/Hospital_Readmission_Part1/Hospital_Readmission_Part1"
$VENV    = "$PROJECT/airflow_venv"
$AIRFLOW_HOME = "$PROJECT/airflow"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Hospital Readmission — Airflow Launcher  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Airflow Standalone inside WSL..." -ForegroundColor Yellow
Write-Host "Open your browser at: http://localhost:8080" -ForegroundColor Green
$pw = Get-Content "$PSScriptRoot\airflow\standalone_admin_password.txt" -ErrorAction SilentlyContinue
if ($pw) {
    Write-Host "Username : admin" -ForegroundColor Green
    Write-Host "Password : $pw" -ForegroundColor Green
}
Write-Host ""
Write-Host "Press Ctrl+C in the WSL window to stop Airflow." -ForegroundColor Yellow
Write-Host ""

# Build the bash command
$bashCmd = @"
export AIRFLOW_HOME=$AIRFLOW_HOME
source $VENV/bin/activate
# Make sure the project src is importable
export PYTHONPATH=$PROJECT:\$PYTHONPATH
cd $PROJECT
airflow standalone
"@

# Launch WSL with the command in a new window so the user can see logs
Start-Process wsl -ArgumentList @("-e", "bash", "-c", $bashCmd)
