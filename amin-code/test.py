from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# پشتیبانی از درخواست‌های CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # آدرس فرانت‌اند را می‌توانید جایگزین کنید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataModel(BaseModel):
    name: str
    age: int

@app.post("/process")
async def process_data(data: DataModel):
    return {"message": f"Hello {data.name}, you are {data.age} years old!"}
