from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
RAW_DIR = ROOT / os.getenv("RAW_DIR", "data/raw")
STAGING_DIR = ROOT / os.getenv("STAGING_DIR", "data/staging")
CLEANED_DIR = ROOT / os.getenv("CLEANED_DIR", "data/cleaned")
ERROR_DIR = ROOT / os.getenv("ERROR_DIR", "data/errors")

for p in [RAW_DIR, STAGING_DIR, CLEANED_DIR, ERROR_DIR]:
    p.mkdir(parents=True, exist_ok=True)

POSTGRES_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'hospital_readmission')}"
)
