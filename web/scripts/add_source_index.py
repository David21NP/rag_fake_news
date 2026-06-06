# NOTE: Comandos actualizar tabla
# ALTER TABLE documents_emba_chunka ADD COLUMN source_index INTEGER;
# ALTER TABLE documents_embb_chunka ADD COLUMN source_index INTEGER;

# UPDATE documents_emba_chunka SET source_index = id - 1;
# UPDATE documents_embb_chunka SET source_index = id - 1;

import pandas as pd
import psycopg2
import psycopg2.extras

import common.utils
from config import get_settings
from rag.utils import build_chunks_sliding


def update_source_index_chunkb():
    settings = get_settings()
    df_train = common.utils.get_df_train()
    current_id = 1
    updates: list[tuple[int, int, int]] = []

    for source_index, row in enumerate(df_train.itertuples(index=False)):
        text = (
            ("" if pd.isna(row.title) else str(row.title))
            + "\n\n"
            + ("" if pd.isna(row.text) else str(row.text))
        )
        n_chunks = len(build_chunks_sliding(text))
        updates.append((source_index, current_id, current_id + n_chunks - 1))
        current_id += n_chunks

    with psycopg2.connect(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
    ) as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                (
                    "UPDATE documents_emba_chunkb SET source_index "
                    "= %s WHERE id BETWEEN %s AND %s"
                ),
                updates,
            )
            psycopg2.extras.execute_batch(
                cur,
                (
                    "UPDATE documents_embb_chunkb SET source_index "
                    "= %s WHERE id BETWEEN %s AND %s"
                ),
                updates,
            )


# UPDATE documents_emba_chunkb SET source_index = %s WHERE id BETWEEN %s AND %s
if __name__ == "__main__":
    update_source_index_chunkb()
