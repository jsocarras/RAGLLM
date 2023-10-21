import streamlit as st
import json
import os
import pinecone
import fitz
import glob
from tqdm.notebook import tqdm
import hashlib
import pickle
from llama_index.text_splitter import SentenceSplitter
from llama_index.schema import TextNode
from llama_index.node_parser.extractors import MetadataExtractor, QuestionsAnsweredExtractor, TitleExtractor
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from dotenv import load_dotenv

print(f"Initializing script...")

# Step 1: Environment Setup
# This line not needed when running in streamlit cloud:
# load_dotenv(dotenv_path=".env")
DATA_DIR = './data'

# Step 2: Pinecone Setup
# api_key = os.environ["PINECONE_API_KEY"]
# environment = os.environ["PINECONE_ENVIRONMENT"]
api_key = st.secrets["PINECONE_API_KEY"]
environment = st.secrets["PINECONE_ENVIRONMENT"]
pinecone.init(api_key=api_key, environment=environment)
index_name = "llamaindex-rag-fs"

existing_indexes = pinecone.list_indexes()
if index_name not in existing_indexes:
    pinecone.create_index(index_name, dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index(index_name)

# Streamlit frontend
st.title("Welcome to the LLM Query Interface!")
st.write("""
Upload your PDF files and query the LLM with ease. Follow the steps below:
1. Upload your PDF files.
2. Query the LLM using the input bar at the bottom.
""")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "json"], accept_multiple_files=True)

if uploaded_files:
    pdf_files = []
    for file in uploaded_files:
        # Ensure DATA_DIR exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        # Save the file to the /data directory
        filepath = os.path.join(DATA_DIR, file.name)
        file_content = file.getvalue()
        with open(filepath, "wb") as f_out:
            f_out.write(file_content)
        pdf_files.append((filepath, file_content))

    text_chunks = []
    doc_idxs = []
    originating_pdf = []

    def generate_content_hash(content):
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.md5(content).hexdigest()

    indexed_hashes_file = "indexed_hashes.pkl"
    if os.path.exists(indexed_hashes_file):
        with open(indexed_hashes_file, 'rb') as f:
            indexed_hashes = pickle.load(f)
    else:
        indexed_hashes = set()

    for filepath, file_content in pdf_files:
        doc_hash = generate_content_hash(file_content)
        if doc_hash in indexed_hashes:
            continue

        indexed_hashes.add(doc_hash)

        # Check the file extension to decide how to process the content
        _, file_extension = os.path.splitext(filepath)
        if file_extension == ".pdf":
            doc = fitz.open(filename=filepath)  # Open using filename, not stream
            text_splitter = SentenceSplitter(chunk_size=1024)
            for doc_idx, page in enumerate(doc):
                page_text = page.get_text("text")
                cur_text_chunks = text_splitter.split_text(page_text)
                text_chunks.extend(cur_text_chunks)
                doc_idxs.extend([doc_idx] * len(cur_text_chunks))
                originating_pdf.extend([file.name] * len(cur_text_chunks))
        elif file_extension == ".txt":
            # For TXT files, split the content into chunks
            text_splitter = SentenceSplitter(chunk_size=1024)
            cur_text_chunks = text_splitter.split_text(file_content.decode("utf-8"))
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([0] * len(cur_text_chunks))  # Use 0 as the default document index for TXT files
            originating_pdf.extend([file.name] * len(cur_text_chunks))
        elif file_extension == ".json":
            # Assuming the JSON has a "content" key that holds the main text
            json_content = json.loads(file_content)
            main_text = json_content.get("content", "")
            text_splitter = SentenceSplitter(chunk_size=1024)
            cur_text_chunks = text_splitter.split_text(main_text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([0] * len(cur_text_chunks))  # Use 0 as the default document index for JSON files
            originating_pdf.extend([file.name] * len(cur_text_chunks))

    with open(indexed_hashes_file, 'wb') as f:
        pickle.dump(indexed_hashes, f)

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        content_hash = generate_content_hash(text_chunk)
        node = TextNode(id=content_hash, text=text_chunk, metadata={"originating_pdf": originating_pdf[idx]})
        nodes.append(node)

    llm = OpenAI(model="gpt-3.5-turbo")
    metadata_extractor = MetadataExtractor(
        extractors=[
            TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
        ],
        in_place=False,
    )
    nodes = metadata_extractor.process_nodes(nodes)

    embed_model = OpenAIEmbedding()
    added_node_ids = set()

    for node in nodes:
        if node.id_ not in added_node_ids:
            node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
            node.embedding = node_embedding
            vector_store.add([node])
            added_node_ids.add(node.id_)

    st.success(f"Processed {len(uploaded_files)} files and updated the index!")

# Text input for querying the LLM
query_input = st.text_input("Enter your query below:")

if query_input:
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine()
    response = query_engine.query(query_input)
    try:
        # Extract and display the originating PDFs from the response
        originating_pdfs_in_response = [node.metadata['originating_pdf'] for node in response.source_nodes if 'originating_pdf' in node.metadata]
        unique_pdfs_in_response = set(originating_pdfs_in_response)
        st.write(f"Response: {str(response)}")
        st.write(f"Response sourced from: {unique_pdfs_in_response}")
    except AttributeError:
        st.write(f"Failed to extract nodes from the response for query: {query_input}")
