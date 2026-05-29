import psycopg2

from config import Settings


def test_db_connection(settings: Settings):
    with psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
