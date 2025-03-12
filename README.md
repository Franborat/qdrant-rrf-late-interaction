# Qdrant Cluster with Dense, Sparse, and Late Interaction Embeddings

This project demonstrates how to **deploy a Qdrant vector database cluster**, **ingest documents with different embedding models**, and **query the stored data using hybrid search and late interaction reranking**.

## **Overview**
The project consists of:
- A **Qdrant cluster** (2-node setup using Docker Compose)
- An **ingestion script (`ingest.py`)** that processes documents and stores embeddings in Qdrant
- A **querying script (`query.py`)** that searches Qdrant using dense, sparse, and late interaction embeddings

---

## **1. Setting Up Qdrant Cluster**
The Qdrant cluster is provisioned using **Docker Compose** with two nodes.

### **Start the Cluster**
Ensure you have **Docker** installed, then run:
```sh
docker-compose up -d
```
This will:
- Start **qdrant_node1** (main node) on port `6333`
- Start **qdrant_node2** (secondary node) and join it to the cluster

To check if the cluster is running:
```sh
docker ps
```

To stop the cluster:
```sh
docker-compose down
```

---

## **2. Installing Dependencies**
Before running the ingestion and query scripts, install the required Python dependencies:
```sh
pip install -r requirements.txt
```
This will install all necessary libraries, including:
- `qdrant-client` (Qdrant database client)
- `fastembed` (embedding model wrapper)

---

## **3. Ingesting Data into Qdrant**
The **`ingest.py`** script:
- Reads documents from `titles_with_users.json`
- Computes **dense, sparse, and late interaction embeddings**
- Stores them in **Qdrant** for later retrieval

### **Run Ingestion**
```sh
python ingest.py
```
This will:
1. **Load models** for embedding generation:
   - `sentence-transformers/all-MiniLM-L6-v2` (Dense embeddings)
   - `prithivida/Splade_PP_en_v1` (Sparse embeddings)
   - `colbert-ir/colbertv2.0` (Late interaction embeddings)
2. **Create a Qdrant collection** (`demo_collection`) if it doesn’t exist
3. **Convert documents into embeddings**
4. **Upload embeddings & metadata to Qdrant**
5. **Optimize indexing** for fast search

---

## **4. Querying Qdrant**
The **`query.py`** script:
- Accepts a search query
- Generates **dense, sparse, and late interaction embeddings**
- Uses **Reciprocal Rank Fusion (RRF)** to combine results
- Applies **late interaction reranking** for final ranking

### **Run a Query**
```sh
python query.py
```
This will:
1. Generate embeddings for `"My query"`
2. Perform a **hybrid search**:
   - **Sparse retrieval** (Splade)
   - **Dense retrieval** (MiniLM)
   - **RRF fusion** to combine rankings
3. **Apply late interaction reranking** (ColBERT) to refine the top results
4. Return the **top 10 most relevant documents**

---

## **5. Project Structure**
```
├── docker-compose.yaml   # Qdrant cluster setup
├── ingest.py             # Script to process and store documents in Qdrant
├── query.py              # Script to search and retrieve documents from Qdrant
├── requirements.txt      # Python dependencies
├── titles_with_users.json # Sample dataset (not included)
```

### **Technologies Used**
- **[Qdrant](https://qdrant.tech/)** - Vector database for similarity search
- **[FastEmbed](https://github.com/quickembedding/fastembed)** - Embedding model wrapper
- **[Docker](https://www.docker.com/)** - Containerized Qdrant setup
- **Python (FastEmbed, Qdrant Client)**

---

## **6. Customization**
### **Change the Collection Name**
Modify `collection_name` in **`ingest.py`** and **`query.py`**.

### **Use a Different Dataset**
Replace `titles_with_users.json` with your own JSON file. Ensure it contains **"title"** fields.

### **Modify the Number of Retrieved Results**
Adjust `limit=10` in **`query.py`** to return more or fewer documents.

---

## **7. Troubleshooting**
### **Qdrant Is Not Running**
Run:
```sh
docker ps
```
If Qdrant is missing, restart it:
```sh
docker-compose up -d
```

### **Connection Issues**
Ensure Qdrant is accessible on `http://localhost:6333`:
```sh
curl http://localhost:6333/healthz
```

### **Performance Optimization**
- Increase **shards** and **replication factor** in `ingest.py`

---

## **8. Next Steps**
- Add **semantic filtering** (e.g., filter by `user_id`)
- Experiment with **different embedding models** for retrieval

🚀 **Enjoy building hybrid search systems with Qdrant!**

