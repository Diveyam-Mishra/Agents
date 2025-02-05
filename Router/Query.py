from fastapi import APIRouter, Depends, HTTPException,BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict,Any
import asyncio
from Controller.Query import process_csv,process_DB
from Schemas.User_input import User_input

router=APIRouter()


@router.post("/upload_csv/{file_path}")
async def NLP_to_CSV_Query(file_path: str, user_input: User_input):
    try:
        file_path = Path(file_path)

        if not file_path.suffix == ".csv":
            raise ValueError("The provided file is not a CSV.")
        
        
        result = await process_csv(file_path, user_input)
        
        return {"message": "Query executed successfully", "result": result}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@router.post("/upload_csv/{file_path}")
async def NLP_to_CSV_Query(file_path: str, user_input: User_input):
    try:
        file_path = Path(file_path)

        if not file_path.suffix == ".csv":
            raise ValueError("The provided file is not a CSV.")
        
        
        result = await process_csv(file_path, user_input)
        
        return {"message": "Query executed successfully", "result": result}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@router.post("/acess_Db/{Client_code}")
async def NLP_to_DB_Query(client_code: str, user_input: User_input):
    try:  
        result = await process_DB(client_code, user_input)
        return {"message": "Query executed successfully", "result": result}
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")