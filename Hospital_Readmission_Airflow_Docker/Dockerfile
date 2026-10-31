FROM apache/airflow:3.3.1

USER airflow

# Project ETL dependencies. Airflow's own dependency stack is kept intact.
RUN pip install --no-cache-dir \
    pandas \
    python-dotenv \
    psycopg2-binary

ENV PYTHONPATH=/opt/airflow/project
