from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.utils import download_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

# --- NEW IMPORTS FOR DENSENET-121 ---
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Flask Setup
# --------------------------------------------------

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------------------------------
# Environment & Setup
# --------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY missing in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in .env")

# --------------------------------------------------
# Load Local Fine-Tuned DenseNet-121 (4 Classes)
# --------------------------------------------------
print("Loading Local Fine-Tuned DenseNet-121 Architecture...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate architecture to match your Kaggle checkpoint parameters
vision_model = models.densenet121()
num_ftrs = vision_model.classifier.in_features
vision_model.classifier = nn.Linear(num_ftrs, 4)  # 4 checkpoint channels

# Load your downloaded weights file
WEIGHTS_PATH = 'densenet121_weights.pth'
if os.path.exists(WEIGHTS_PATH):
    vision_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"DenseNet-121 weights successfully loaded from {WEIGHTS_PATH}")
else:
    print(f"WARNING: {WEIGHTS_PATH} not found in root directory! Please copy it here.")

vision_model.to(device)
vision_model.eval()

# Image Preprocessing Pipeline
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================================
# CRITICAL RULE: REPLACE THESE 4 STRINGS WITH THE EXACT MEDICAL NAMES
# OF YOUR KAGGLE DATASET FOLDERS (e.g., "COVID-19", "Pneumonia", "Normal")
# =========================================================================
CLASS_MAPPING = {
    0: "COVID-19 Infection",
    1: "Normal Lung Structure",
    2: "Pneumonia",
    3: "Tuberculosis"
}

def analyze_medical_image(image_path):
    """Processes an uploaded image through DenseNet-121 and returns its prediction."""
    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = img_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = vision_model(input_tensor)
            _, preds = torch.max(outputs, 1)
            
        return CLASS_MAPPING[preds.item()]
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "Unknown Medical Finding"

# --------------------------------------------------
# Embeddings & Pinecone
# --------------------------------------------------

print("Loading Embedding Model...")
embeddings = download_embeddings()
print("Embedding Model Loaded Successfully")

index_name = "medical-research"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)

# --------------------------------------------------
# Groq LLM & Prompt Engineering
# --------------------------------------------------

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
    temperature=0
)

# Standard template schema supporting dynamic image variables inside user metadata context
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Additional Image Finding Context: {image_context}\n\nUser Question: {input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    image_file = request.files.get("image")  # Capture file input from Form Data
    
    if not msg and not image_file:
        return "Please enter a question or upload an image."

    print("\n==============================")
    print("USER TEXT MESSAGE:", msg)
    print("==============================")

    image_finding = None
    image_context_string = "No image uploaded by user."

    # Process Image if uploaded
    if image_file and image_file.filename != '':
        filename = secure_filename(image_file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(temp_filepath)
        
        print(f"Processing uploaded image: {filename}")
        image_finding = analyze_medical_image(temp_filepath)
        print(f"DenseNet-121 Diagnostic Output: {image_finding}")
        
        image_context_string = f"The user has uploaded a chest X-ray. Your local imaging vision system (DenseNet-121) analyzed it and flagged the findings as: {image_finding}."
        
        # Cleanup file after analysis
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

    try:
        # Determine the query string we use to pull files from Pinecone
        # If user typed nothing, look up the medical condition text directly in books!
        search_query = msg if msg else f"Clinical guidelines and treatment for {image_finding}"
        
        print(f"Querying Pinecone database for: '{search_query}'")
        docs = retriever.invoke(search_query)

        print("\n===== RETRIEVED DOCUMENTS =====")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            print(f"\nDOCUMENT {i+1} (source: {source})")
            print(doc.page_content[:400])
            print("\n------------------------------")

        # Fallback if text chunks completely failed to find keywords in your specific vector database
        if not docs or len(docs) == 0:
            if image_finding:
                return f"Vision Analysis Result: **{image_finding}**. (Note: No matching textbook materials were found in your Pinecone index regarding this specific term.)"

        # Invoke dynamic multimodal context RAG pipeline
        response = rag_chain.invoke({
            "input": msg if msg else f"Provide clinical information explaining the nature, symptoms, and management of: {image_finding}",
            "image_context": image_context_string
        })

        print("\n===== ANSWER =====")
        print(response["answer"])

        return str(response["answer"])

    except Exception as e:
        print("\n===== ERROR =====")
        print(repr(e))
        return "Sorry, something went wrong while generating the answer. Please try again."

# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )