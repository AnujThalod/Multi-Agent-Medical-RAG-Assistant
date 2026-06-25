Multi-Agent Medical RAG Chatbot

A multimodal medical AI assistant that combines a Retrieval-Augmented Generation (RAG) chatbot with a fine-tuned DenseNet-121 chest X-ray classifier. Users can ask medical questions in text, upload a chest X-ray image, or speak their question aloud — the system routes the image through a CNN classifier, then feeds the diagnostic finding into the RAG pipeline so the LLM can generate a grounded, document-backed explanation.


⚠️ Educational project only. This is not a medical device and must never be used for real diagnosis or treatment decisions. Always consult a licensed physician or radiologist.




What it does


Text-based medical Q&A — ask general medical questions, answered using a Pinecone-indexed knowledge base of medical reference documents, via a Groq-hosted LLM.
Chest X-ray classification — upload an X-ray image (drag-and-drop or click-to-upload) and a locally fine-tuned DenseNet-121 model classifies it into one of four categories: COVID-19, Normal, Pneumonia, Tuberculosis.
Multimodal explanation — the CNN's finding is passed into the RAG chain as additional context, so the LLM explains the result using relevant retrieved medical documents, instead of just returning a bare label.
Voice input — ask questions by speaking, using the browser's built-in speech recognition (Chrome/Edge).



Architecture

                    ┌─────────────────────┐
                    │     Chat UI (HTML)   │
                    │  text / image / mic  │
                    └──────────┬───────────┘
                               │ POST /get
                               ▼
                    ┌─────────────────────┐
                    │     Flask Backend    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴─────────────────┐
              ▼                                   ▼
   ┌─────────────────────┐           ┌─────────────────────────┐
   │  DenseNet-121 (CNN)   │           │   Pinecone Retriever     │
   │  Image classification │           │   (medical document       │
   │  4 classes:            │           │    embeddings)            │
   │  COVID-19 / Normal /   │           └────────────┬─────────────┘
   │  Pneumonia / TB        │                        │
   └───────────┬────────────┘                        │
               │ image_finding                       │ retrieved context
               └───────────────────┬──────────────────┘
                                   ▼
                       ┌─────────────────────┐
                       │   Groq LLM (RAG)      │
                       │   generates grounded   │
                       │   explanation           │
                       └───────────┬─────────────┘
                                   ▼
                          Response to user

When an image is uploaded, the classifier's output (e.g. "COVID-19 Infection") is injected into the prompt as image_context, alongside whatever relevant chunks the retriever pulls from the medical document index — so the LLM's explanation is grounded in both the visual finding and the reference material, not just one or the other.


Tech stack

LayerTechnologyBackendFlaskLLMGroq (openai/gpt-oss-120b) via langchain-groqVector storePinecone, via langchain-pineconeRAG orchestrationLangChainImage classifierPyTorch, DenseNet-121 (fine-tuned, 4-class)FrontendHTML, vanilla JS, Bootstrap 5Voice inputWeb Speech API (browser-native)


Project structure

.
├── app.py                  # Flask app: routes, CNN inference, RAG chain
├── requirements.txt
├── .env.example             # Template for required environment variables
├── .gitignore
├── src/
│   ├── prompt.py            # System prompt for the LLM
│   └── utils.py             # Embedding model loader
├── templates/
│   └── chat.html            # Chat UI (text, image upload, drag-and-drop, voice)
└── static/
    └── chat.css             # Styling


Setup

1. Clone the repo

bashgit clone https://github.com/YOUR_USERNAME/Multi-Agent-Medical-RAG-Chatbot.git
cd Multi-Agent-Medical-RAG-Chatbot

2. Create a virtual environment and install dependencies

bashpython -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Set up environment variables

Copy the example file and fill in your own keys:

bashcp .env.example .env

PINECONE_API_KEY=your_pinecone_key_here
GROQ_API_KEY=your_groq_key_here

4. Download the model weights

The fine-tuned DenseNet-121 weights are too large for this repository and are hosted on Hugging Face Hub:

👉 Download densenet121_weights.pth from Hugging Face Hub
(replace this link with your actual Hugging Face model page URL)

Place the downloaded file in the project root, alongside app.py:

.
├── app.py
├── densenet121_weights.pth   ← place it here
└── ...

5. Set up your Pinecone index

This project expects a Pinecone index named medical-research populated with embedded medical reference documents. Index your own document set using the embedding pipeline in src/utils.py before running the app, if you haven't already.

6. Run the app

bashpython app.py

Visit http://localhost:8080 in your browser (Chrome or Edge recommended, for voice input support).


Usage


Ask a text question — type into the input box and press Enter or click send.
Upload an X-ray — click the paperclip icon, or drag and drop an image file anywhere onto the chat window.
Speak a question — click the microphone icon, speak, and the transcribed text fills the input box automatically.
Combine both — upload an X-ray and add a specific question (e.g. "what treatment is recommended for this?") to get a more targeted explanation.



Model details


Architecture: DenseNet-121, fine-tuned (final classifier layer replaced for 4-class output).
Classes: COVID-19 Infection, Normal Lung Structure, Pneumonia, Tuberculosis.
Training data: chest X-ray images sourced from Kaggle, with significant class imbalance across the 4 categories.
Imbalance handling: class-weighted cross-entropy loss, with weights inversely proportional to class frequency.



Known limitations & future work


Class imbalance — the training set is unevenly distributed across the 4 classes, and class weighting alone may not fully resolve minority-class performance. Planned improvement: evaluate per-class precision/recall via a confusion matrix and consider data augmentation or resampling for underrepresented classes.
Voice input is browser-dependent — uses the Web Speech API, which is well-supported in Chrome/Edge but not in Firefox and only partially in Safari. A more robust version would use a server-side speech-to-text API (e.g. Groq's Whisper endpoint).
Not a diagnostic tool — the CNN classifier is a portfolio/educational demonstration, not a clinically validated model. It has not been evaluated against clinical-grade benchmarks or reviewed by medical professionals.
Single-image inference only — no support yet for multi-view X-rays (e.g. PA + lateral) or DICOM format, which real clinical workflows typically use.



Disclaimer

This project is for educational and portfolio purposes only. It is not intended for, and must not be used for, real clinical diagnosis, treatment decisions, or any other medical use. Always consult a qualified healthcare professional for medical concerns.