import os
import time
import json
from redis import Redis
from rq import Queue, Connection, Worker
from ingestion.parse_utils import extract_text_by_type
import psycopg2
from psycopg2.extras import RealDictCursor


def get_redis():
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return Redis.from_url(redis_url)


def get_db_conn():
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "app")
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return psycopg2.connect(dsn)


def process_parse_job(job_id, resource_id, storage_path):
    # Mark job as processing
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE job SET status=%s, updated_at=now() WHERE id=%s", ("processing", job_id))
                conn.commit()
        finally:
            conn.close()
    except Exception:
        # best-effort; continue
        pass

    try:
        # storage_path: bucket/object
        # try to download from MinIO if available, otherwise look in sample/
        local_path = None
        sample_dir = os.path.join(os.getcwd(), "sample")
        fname = storage_path.split("/")[-1]
        if os.path.isdir(sample_dir):
            for root, dirs, files in os.walk(sample_dir):
                if fname in files:
                    local_path = os.path.join(root, fname)
                    break

        if not local_path:
            # try downloading from MinIO
            try:
                from minio import Minio
                m = Minio(os.getenv("MINIO_ENDPOINT", "minio:9000"), access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"), secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"), secure=os.getenv("MINIO_SECURE", "false").lower() in ("1","true","yes"))
                bucket, obj = storage_path.split("/", 1)
                tmpfile = os.path.join(os.getcwd(), "tmp_" + obj)
                m.fget_object(bucket, obj, tmpfile)
                local_path = tmpfile
            except Exception:
                local_path = None

        pages = [""]
        if local_path:
            pages = extract_text_by_type(local_path, None)

        # run math extraction on PDF images (pix2tex) and map by page
        try:
            from ingestion.math_extractor import extract_math_from_pdf
            math_map = {}
            if local_path and local_path.lower().endswith('.pdf'):
                math_map = extract_math_from_pdf(local_path)
        except Exception:
            math_map = {}

        conn = get_db_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # insert pages
                for i, p in enumerate(pages, start=1):
                    cur.execute("INSERT INTO extracted_page (resource_id, page_number, raw_text, created_at) VALUES (%s,%s,%s,now())",
                                (resource_id, i, p))
                    # optionally insert math expressions as separate rows or attach to chunks later
                    if i in math_map:
                        # store as a simple extracted_page update in this MVP
                        cur.execute("UPDATE extracted_page SET raw_text = raw_text || %s WHERE resource_id=%s AND page_number=%s",
                                    ("\n\nMATH:\n" + "\n\n".join(math_map[i]), resource_id, i))
                cur.execute("UPDATE job SET status=%s, updated_at=now() WHERE id=%s", ("done", job_id))
                conn.commit()
        finally:
            conn.close()
    except Exception:
        # On any failure, mark job as error
        try:
            conn = get_db_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("UPDATE job SET status=%s, updated_at=now() WHERE id=%s", ("error", job_id))
                    conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    redis = get_redis()
    q = Queue("parse", connection=redis)
    with Connection(redis):
        worker = Worker([q], connection=redis)
        worker.work()


