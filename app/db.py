from sqlalchemy import create_engine, text
import pandas as pd
from config import DATABASE_URL

# Create database engine
engine = create_engine(DATABASE_URL)

# ---------------- INSERT ----------------
def insert_job_application(company, role, applied_date, website, status, notes):
    query = text("""
        INSERT INTO job_applications (company_name, job_role, applied_date, website, status, notes)
        VALUES (:company, :role, :applied_date, :website, :status, :notes)
    """)
    
    with engine.begin() as conn:  # auto-commit
        conn.execute(query, {
            "company": company,
            "role": role,
            "applied_date": applied_date,
            "website": website,
            "status": status,
            "notes": notes
        })

# ---------------- FETCH ALL ----------------
def get_all_applications():
    query = text("SELECT * FROM job_applications ORDER BY applied_date DESC")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

# ---------------- UPDATE STATUS ----------------
def update_status(app_id, new_status):
    query = text("""
        UPDATE job_applications
        SET status = :status
        WHERE id = :app_id
    """)
    with engine.begin() as conn:
        conn.execute(query, {"status": new_status, "app_id": app_id})

# ---------------- DELETE ----------------
def delete_application(app_id):
    query = text("DELETE FROM job_applications WHERE id = :app_id")
    with engine.begin() as conn:
        conn.execute(query, {"app_id": app_id})


































