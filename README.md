# MultiModal-Video-Image-Search-with-Florence

How It Works
Keyframe Extraction:
For each video, keyframes are extracted at a regular interval (e.g., every 30 frames). These frames are converted into a PIL image for consistency.

Embedding Generation:
Each image and keyframe is preprocessed and passed through Florence’s image encoder. Similarly, text queries are tokenized and converted to embeddings using Florence’s text encoder.

Indexing with FAISS:
The resulting embeddings are stored in a FAISS index (using L2 distance), allowing efficient nearest neighbor retrieval across large datasets.

Query Execution:
When a query is made (text or image), its embedding is computed, and the FAISS index is used to quickly find the most similar embeddings, returning associated image or video frame metadata.

