from fastapi import APIRouter, Depends, HTTPException,BackgroundTasks,UploadFile, File
from pydantic import BaseModel
from typing import List, Dict,Any
import pandas as pd
from io import StringIO, BytesIO
from Sales_Prediction.Model.Input_validator import DataValidator
from typing import Dict, Any
import lightgbm as lgb



class InputController:
    def __init__(self):
        self.validator = DataValidator()
        
    async def process_files(
        self,
        background_tasks: BackgroundTasks,
        stock_file: UploadFile,
        item_master: UploadFile
    ) -> Dict[str, Any]:
    
        if not stock_file.filename.endswith(('.csv','.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="Stock file must be Excel format")
        if not item_master.filename.endswith(('.csv','.xls', '.xlsx')):
            raise HTTPException(status_code=400, detail="Item master must be CSV format")

        stock_df = await self.validator.process_stock_file(stock_file)
        item_df = await self.validator.process_item_master(item_master)
        
        cleaned_data = self.validator.clean_data(stock_df, item_df)
        
        background_tasks.add_task(self.validator.train_model)
    
        response = {
            "message": "Files processed successfully",
            "stock_file": stock_file.filename,
            "item_master": item_master.filename,
            "record_count": len(cleaned_data),
            "columns": list(cleaned_data.columns),
            "training_status": "Scheduled for background processing"
        }
        
        return response

    async def get_data(self) -> Dict[str, Any]:
        if self.validator.cleaned_data is None:
            raise HTTPException(status_code=404, detail="No processed data available")
            
        return {
            "record_count": len(self.validator.cleaned_data),
            "columns": list(self.validator.cleaned_data.columns),
            "sample_data": self.validator.cleaned_data.head(5).to_dict('records')
        }

