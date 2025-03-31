from multiprocessing import process
import subprocess
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
import os
import json
from typing import Optional
import shutil
from utils.question_matching import find_similar_question
from utils.file_process import process_uploaded_file
from utils.function_definations_llm import function_definitions_objects_llm
from utils.openai_api import extract_parameters
from utils.solution_functions import functions_dict

tmp_dir = "tmp_uploads"
os.makedirs(tmp_dir, exist_ok=True)

app = FastAPI()


@app.get("/")
def fun():
    return "works"

SECRET_PASSWORD = os.getenv("SECRET_PASSWORD")

# Modify the redeploy endpoint
@app.get('/redeploy')
def redeploy(password: str = Query(None)):
    if password != SECRET_PASSWORD:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    # Replace shell script execution with a message
    return {"message": "Redeployment not available in cloud environment"}


async def save_upload_file(upload_file: UploadFile) -> str:
    """Save an uploaded file to disk and return its path"""
    file_path = os.path.join(tmp_dir, upload_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return file_path


@app.post("/api")
async def process_file(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    file_names = []
    tmp_dir_local = tmp_dir  # Initialize tmp_dir_local with the global tmp_dir value

    # Handle the file processing if file is present
    matched_function = find_similar_question(
        question
    )  # Function to compare using cosine similarity
    
    # Extract just the function name from the tuple
    function_name = matched_function[0]
    print("-----------Matched Function------------\n", function_name)
    
    if file:
        # Save and process the uploaded file (ZIP or image)
        file_path = await save_upload_file(file)
        try:
            tmp_dir_local, file_names = process_uploaded_file(file_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Extract parameters using the matched function
    parameters = extract_parameters(
        str(question),
        function_definitions_llm=function_definitions_objects_llm.get(function_name, {}),
    )  # Function to call OpenAI API and extract parameters

    print("-----------parameters------------\n", parameters)

    # Validate if parameters were extracted successfully
    if not parameters or "arguments" not in parameters:
        raise HTTPException(
            status_code=400, 
            detail="Failed to extract parameters for the given question"
        )

    solution_function = functions_dict.get(
        function_name, lambda **kwargs: "No matching function found"
    )  # the solutions functions name is same as in questions.json

    # Parse the arguments from the parameters
    try:
        arguments = json.loads(parameters["arguments"])
    except (TypeError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid arguments format: {str(e)}"
        )

    print("-----------arguments------------\n", arguments)

    # For compress_an_image, override the image_path with the actual path
    if matched_function == "compress_an_image" and file_names and tmp_dir_local:
        # Use the first uploaded image file
        actual_image_path = os.path.join(tmp_dir_local, file_names[0])
        arguments["image_path"] = actual_image_path
        print(f"Overriding image path to: {actual_image_path}")

    # Call the solution function with the extracted arguments
    try:
        answer = solution_function(**arguments)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error executing function: {str(e)}"
        )

    # Return the answer in JSON format
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
