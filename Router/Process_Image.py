from fastapi import APIRouter, File, UploadFile, HTTPException
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
load_env = os.getenv("ENVIRONMENT", "development")

router=APIRouter()

AWS_ACCESS_KEY = os.environ.get()
AWS_SECRET_KEY = os.environ.get()
AWS_S3_BUCKET = os.environ.get()
AWS_S3_REGION = os.environ.get()    

DB_HOST = os.environ.get("DB_HOST")    
DB_PORT = os.environ.get("DB_PORT")    
DB_NAME = os.environ.get("DB_NAME")    
DB_USER = os.environ.get("DB_USER")    
DB_PASSWORD = os.environ.get("DB_PASSWORD")    


s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_S3_REGION,
)

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        return conn, cur
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    valid_image_types = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp", "image/tiff"}
    if file.content_type not in valid_image_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        filename = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"

        s3_client.upload_fileobj(
            file.file, AWS_S3_BUCKET, filename, ExtraArgs={"ContentType": file.content_type}
        )
        s3_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_S3_REGION}.amazonaws.com/{filename}"

        conn, cur = connect_to_db()

        cur.execute(
            """
            INSERT INTO images (filename, content_type, file_size, s3_url)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """,
            (file.filename, file.content_type, file.spool_max_size, s3_url),
        )
        image_id = cur.fetchone()["id"]
        conn.commit()
        cur.close()
        conn.close()

        return {
            "message": "Image uploaded successfully",
            "image_id": image_id,
            "s3_url": s3_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

