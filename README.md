# MultiModal-Video-Image-Search-with-Florence

Install dependencies via pip:

pip install torch opencv-python faiss-cpu numpy pillow scikit-learn
Usage
Clone the Repository:
git clone https://github.com/yourusername/multimodal-video-image-search.git
cd multimodal-video-image-search
Set Up Florence: Ensure you have the correct version of Florence loaded. Adjust the model loading and preprocessing functions if necessary based on the Florence documentation.

Index Your Media: Update the paths for your image and video directories in the script. The tool will:

For images: Preprocess and compute embeddings.
For videos: Extract keyframes (every 30 frames by default) and compute embeddings for each frame.
Perform a Search: Use the provided query functions to encode your text or image queries. The system computes similarity scores using FAISS and returns the top matching images and video frames.

Run the Script:
python search_tool.py

How It Works
Keyframe Extraction:
For each video, keyframes are extracted at a regular interval (e.g., every 30 frames). These frames are converted into a PIL image for consistency.

Embedding Generation:
Each image and keyframe is preprocessed and passed through Florence’s image encoder. Similarly, text queries are tokenized and converted to embeddings using Florence’s text encoder.

Indexing with FAISS:
The resulting embeddings are stored in a FAISS index (using L2 distance), allowing efficient nearest neighbor retrieval across large datasets.

Query Execution:
When a query is made (text or image), its embedding is computed, and the FAISS index is used to quickly find the most similar embeddings, returning associated image or video frame metadata.

