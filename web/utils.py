import psycopg2

from config import get_settings


def test_db_connection():
    with psycopg2.connect(
        dbname=get_settings().db_name,
        user=get_settings().db_user,
        password=get_settings().db_password,
        host=get_settings().db_host,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
