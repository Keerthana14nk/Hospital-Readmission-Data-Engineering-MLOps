# Hospital Readmission Analytics – Dockerized Airflow

This package adds a Dockerized Apache Airflow environment for the existing `Hospital_Readmission_Part1` project.

## What is included

- Apache Airflow 3.3.1 Docker image
- PostgreSQL + Redis services for Airflow itself
- `hospital_readmission_part1` DAG
- Four tasks: ingest → clean → validate → load
- Project source mounted at `/opt/airflow/project`
- Windows PostgreSQL reached from containers using `host.docker.internal`
- Airflow UI exposed at `http://localhost:8081`

## Put this folder inside the existing project

The expected layout is:

Hospital_Readmission_Part1/
- docker-airflow/
- src/
- data/
- dashboard/
- airflow/ ...

Copy the files from this package into the project, or rename this folder to `docker-airflow` inside the project.

## First setup

1. Make sure Docker Desktop is running.
2. Make sure the project's `.env` exists one level above this `docker-airflow` folder and contains the PostgreSQL password.
3. Open PowerShell in the `docker-airflow` folder.
4. Run:

```powershell
docker compose build
docker compose up -d
```

5. Open:

http://localhost:8081

Login:

- Username: `admin`
- Password: `Airflow@123`

6. In Airflow, open `hospital_readmission_part1`, unpause it, and trigger it.

## Important

The Airflow PostgreSQL container is only Airflow's metadata database. The project ETL is configured to connect to the existing Windows PostgreSQL database through `host.docker.internal:5432`.

This is a local classroom/development setup, not a production deployment.
