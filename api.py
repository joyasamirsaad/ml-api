from fastapi import FastAPI

# creating instance of the FastAPI application
app = FastAPI()

# GET Request
@app.get("/frame") # api endpoint
async def read_root():
    return {"message": "Hello, FastAPI!"}