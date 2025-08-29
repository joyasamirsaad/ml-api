from fastapi import FastAPI
from api.controllers import api_controller
import uvicorn

app = FastAPI(title="Onion Architecture Example")

# Include routes from the API controller
app.include_router(api_controller.router, prefix="/api", tags=["API"])
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)