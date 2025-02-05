from fastapi import APIRouter, Depends, HTTPException,BackgroundTasks, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from Controller.Input import InputController
from typing import List, Dict,Any
import shutil


router=APIRouter()

input_controller = InputController()
TEMP_DIR = Path("./temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
@router.get("/supported-formats")
async def get_supported_formats() -> Dict[str, List[str]]:
    
    return {
        "supported_formats": [".csv", ".xlsx", ".xls"],
        "description": "Upload CSV or Excel files to process data"
    }


@router.post("/upload/{client}")
async def upload_files(
    client:str,
    background_tasks: BackgroundTasks,
    stock_file: UploadFile,
    item_master: UploadFile
) -> Dict[str, Any]:
    stock_file_path = TEMP_DIR / f"{client}_stock_{stock_file.filename}"
    item_master_file_path = TEMP_DIR / f"{client}_item_master_{item_master.filename}"
    
    with stock_file_path.open("wb") as f:
        shutil.copyfileobj(stock_file.file, f)

    with item_master_file_path.open("wb") as f:
        shutil.copyfileobj(item_master.file, f)

    file_queue.append(stock_file_path)
    stock_file.file.close()
    item_master.file.close()
    
    if client != "Saree":
        
        return await input_controller.process_files(
            background_tasks,
            stock_file_path,
            item_master_file_path
        )
    else:
        return {"message": "Client yet to be added"}

@router.get("/processed-data")
async def get_processed_data() -> Dict[str, Any]:
    return await input_controller.get_data()