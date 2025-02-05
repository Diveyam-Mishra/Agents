from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from Router.Input import router as input_router
from Router.Query import router as csv_query_router


app = FastAPI(title="Automating Training Data Preparation", description="This API is used to automate the process of training data preparation")

origins = ['http://localhost:3000']

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def read_root():
    return {"SupplyMint"}

app.include_router(input_router)
app.include_router(csv_query_router)
