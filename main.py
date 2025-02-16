from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from llamahelper import LlamaHelper
import os
import tempfile

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize components
llama_helper = LlamaHelper()

@app.get("/")
async def get():
	with open("static/index.html", "r") as f:
		return HTMLResponse(content=f.read())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
	if file.filename.split('.')[-1].lower() != 'txt':
		raise HTTPException(status_code=400, detail="Only .txt files are allowed")

	try:
		# Create a temporary file
		tmp = tempfile.TemporaryFile()
		
		# Copy file contents to temporary file
		contents = await file.read()
		tmp.write(contents)
		tmp.seek(0)

		# Process the file
		file_path = os.path.join("uploads", file.filename)
		os.makedirs("uploads", exist_ok=True)
		
		# Save to uploads directory
		with open(file_path, "wb") as f:
			f.write(tmp.read())
		
		# Process and store in ChromaDB
		llama_helper.process_and_store_documents(file_path)
		
		return {"status": "success", "filename": file.filename}
	
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
	finally:
		tmp.close()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	while True:
		query = await websocket.receive_text()
		relevant_chunks = llama_helper.search_relevant_chunks(query)
		response = llama_helper.generate_response(query, relevant_chunks)
		await websocket.send_text(response)

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
