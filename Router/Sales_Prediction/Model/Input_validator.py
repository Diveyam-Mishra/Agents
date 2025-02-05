from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import pandas as pd
from io import StringIO, BytesIO
from datetime import datetime
import asyncio


REQUIRED_COLUMNS = {
    'stock_file': ['ADMSITE_CODE', 'ICODE', 'QTY', 'ENTDT'],
    'item_master': ['itemcode']
}

class DataValidator:
    def __init__(self):
        self.cleaned_data = None
        
    def validate_columns(self, df: pd.DataFrame, file_type: str) -> bool:
        required_cols = REQUIRED_COLUMNS[file_type]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns in {file_type}: {missing_cols}"
            )
        return True

    async def process_file(self, file: UploadFile, file_type: str) -> pd.DataFrame:
        """
        Generalized method to process both CSV and Excel files.
        """
        try:
            contents = await file.read()
            
            # Attempt to read as Excel
            try:
                bytes_io = BytesIO(contents)
                df = pd.read_excel(bytes_io, sheet_name=None)
                combined_df = pd.concat([df[sheet] for sheet in df], ignore_index=True)
            except Exception:
                # If Excel processing fails, process as CSV
                string_io = StringIO(contents.decode('utf-8'))
                combined_df = pd.read_csv(string_io, delimiter='|', low_memory=False)
            
            # Validate columns
            self.validate_columns(combined_df, file_type)
            return combined_df
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing {file_type} file: {str(e)}")

    async def process_stock_file(self, file: UploadFile) -> pd.DataFrame:
        """
        Process stock files, supporting both CSV and Excel formats.
        """
        return await self.process_file(file, 'stock_file')

    async def process_item_master(self, file: UploadFile) -> pd.DataFrame:
        """
        Process item master files, supporting both CSV and Excel formats.
        """
        return await self.process_file(file, 'item_master')

    def clean_data(self, stock_df: pd.DataFrame, item_df: pd.DataFrame) -> pd.DataFrame:
        try:
            stock_df['QTY'] = stock_df['QTY'].abs()
            stock_df['ENTDT'] = pd.to_datetime(stock_df['ENTDT'])
            stock_df = stock_df.rename(columns={'ICODE': 'icode'})

            item_df.dropna(axis=1, how='all', inplace=True)
            item_df['icode'] = item_df['itemcode']
            
            merged_df = pd.merge(stock_df, item_df, on='icode')
            
            self.cleaned_data = merged_df
            return merged_df
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error cleaning data: {str(e)}")

    async def train_model(self):
        """
        Placeholder for model training logic.
        """
        try:
            if self.cleaned_data is None:
                raise ValueError("No cleaned data available for training")
            
            print(f"Model training started at {datetime.now()}")
            await asyncio.sleep(2)
            print(f"Model training completed at {datetime.now()}")
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")