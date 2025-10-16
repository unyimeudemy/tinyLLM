from fastapi import FastAPI
from pydantic import BaseModel
from llm_1.llm_1 import train_llm, infer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles



app = FastAPI()

origins = [
    "http://localhost:5173", 
    "http://46.62.212.74:8000" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],            
)

@app.get("/api/check")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/api/train")
def train_model():
    train_llm()
    return {"message": "Training successful"}


class Query(BaseModel):
    message: str


@app.post("/api/infer")
def infer_model(q: Query):
    res = infer(q.message, 100)
    return {"message": res}



app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
