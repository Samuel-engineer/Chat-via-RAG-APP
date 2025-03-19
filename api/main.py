from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record, deleteAll_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma, deleteAll_doc_from_chroma
import os
import uuid
import logging
import shutil

current_dir = os.path.dirname(__file__)  # Récupère le dossier du script
output_file = os.path.join(current_dir, "result.txt")

logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI()

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']
    
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

#route pour mettre à jour les données dans la base de donnée 

@app.post("/upload-web-doc")
def upload_documents():
    try:
        # Ajoute une trace pour afficher si l'insertion a bien fonctionné
        file_id = insert_document_record("html epo web sites")
        
        # Ajoute des logs pour déboguer
        print(f"Insertion réussie, file_id: {file_id}")

        # Appel asynchrone à index_web_doc_to_chroma
        success = index_document_to_chroma(output_file, file_id)
        
        print(f"Indexation réussie: {success}")

        if success:
            return {"message": "Files html have been successfully uploaded and indexed."}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail="Failed to index html epo web sites")
    except Exception as e:
        # Capture l'exception et renvoie un message détaillé
        print(f"Une erreur s'est produite : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
    finally:
        print("Vérification terminée")

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
    
@app.post("/deleteAll-doc")
def deleteAll_document():
    # Delete from Chroma
    chroma_delete_success = deleteAll_doc_from_chroma()

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = deleteAll_document_record()
        if db_delete_success:
            return {"message": f"Successfully deleted documents from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete documents from the database."}
    else:
        return {"error": f"Failed to delete documents from Chroma."}
