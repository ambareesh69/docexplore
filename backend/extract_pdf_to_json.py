import os
import json
import numpy as np
import argparse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION ---
MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformer model name
NUM_TOPICS = 3                 # Desired number of topics/clusters
SIMILARITY_THRESHOLD = 0.75    # Minimum cosine similarity to assign a topic to a doc

# --- FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""
    try:
        reader = PdfReader(pdf_path)
        print(f"Successfully opened PDF: {pdf_path}")
        print(f"Number of pages: {len(reader.pages)}")
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    text = ""
    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from page {i}: {e}")
    return text

def split_into_paragraphs(text):
    """Split text into paragraphs (chunks) by double newlines."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Split text into {len(paragraphs)} paragraphs")
    return paragraphs

def compute_embeddings(paragraphs, model):
    """Compute embeddings for a list of paragraphs."""
    print(f"Computing embeddings for {len(paragraphs)} paragraphs")
    embeddings = model.encode(paragraphs)
    print(f"Computed embeddings with shape: {embeddings.shape}")
    return embeddings

def perform_clustering(embeddings, num_clusters):
    """
    Cluster the embeddings using KMeans.
    If the number of samples is less than the requested clusters, use the number of samples.
    """
    n_samples = embeddings.shape[0]
    actual_clusters = min(num_clusters, n_samples) if n_samples > 0 else 1
    print(f"Performing clustering with {actual_clusters} clusters")
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_
    print(f"Clustering complete. Labels: {np.unique(labels, return_counts=True)}")
    return labels, centroids

def assign_topics_to_doc(embedding, centroids, threshold):
    """
    Compute cosine similarity between a doc embedding and each centroid.
    Return a list of (topic_id, similarity) for each centroid that meets the threshold.
    """
    similarities = []
    for i, centroid in enumerate(centroids):
        sim = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8)
        similarities.append(sim)
    return [(i, round(sim, 2)) for i, sim in enumerate(similarities) if sim >= threshold]

def extract_topic_keyword(paragraphs):
    """
    Use TF-IDF to determine the top keyword for the given list of paragraphs.
    Returns the top keyword or a default value if no keyword is found.
    """
    if not paragraphs:
        return "NoData"
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    features = vectorizer.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    if scores.size > 0:
        top_index = int(np.argmax(scores))
        return features[top_index]
    else:
        return "Topic"

# --- MAIN SCRIPT ---

def process_pdf_to_json(input_pdf, output_json):
    """Process a PDF file and save the results as JSON."""
    print(f"Processing PDF: {input_pdf}")
    print(f"Output will be saved to: {output_json}")
    
    # 1. Extract text from PDF
    text = extract_text_from_pdf(input_pdf)
    if not text:
        print("No text extracted from PDF.")
        return False

    # 2. Split text into paragraphs
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        print("No paragraphs found.")
        return False

    # 3. Load SentenceTransformer model and compute embeddings
    try:
        print(f"Loading SentenceTransformer model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        embeddings = compute_embeddings(paragraphs, model)
    except Exception as e:
        print(f"Error loading model or computing embeddings: {e}")
        return False

    # 4. Cluster the embeddings into topics
    try:
        labels, centroids = perform_clustering(embeddings, NUM_TOPICS)
        actual_topics = len(np.unique(labels))  # Actual number of clusters
    except Exception as e:
        print(f"Error during clustering: {e}")
        return False

    # 5. Build the "topics" array.
    topics = []
    for i in range(actual_topics):
        cluster_paragraphs = [paragraphs[j] for j in range(len(paragraphs)) if labels[j] == i]
        keyword = extract_topic_keyword(cluster_paragraphs)
        topics.append({
            "topic_id": i,
            "topic": keyword,
            "subtopic": f"Subtopic of {keyword}",
            "count": len(cluster_paragraphs)
        })

    # 6. Build the "docs" array and "matches" array.
    docs = []
    matches = []
    for j, para in enumerate(paragraphs):
        try:
            doc_embedding = embeddings[j]
            assigned = assign_topics_to_doc(doc_embedding, centroids, SIMILARITY_THRESHOLD)
            cluster_label = labels[j]
            centroid = centroids[cluster_label]
            score = np.dot(doc_embedding, centroid) / (np.linalg.norm(doc_embedding) * np.linalg.norm(centroid) + 1e-8)
            score = round(score, 2)
            doc_obj = {
                "file name": os.path.splitext(os.path.basename(input_pdf))[0],
                "section": "Document Section",
                "para": para,
                "chapter": "Document Chapter",
                "sector": "Document Category",
                "cluster": int(cluster_label),
                "score": score,
                "topics": [topic_id for topic_id, sim in assigned]
            }
            docs.append(doc_obj)
            for topic_id, sim in assigned:
                matches.append({
                    "doc": j,
                    "topic": topic_id,
                    "similarity": sim
                })
        except Exception as e:
            print(f"Error processing paragraph {j}: {e}")
            return False

    # 7. Build the metadata
    metadata = {
        "total_chunks": len(paragraphs),
        "embedding_dimension": int(embeddings.shape[1]),
        "total_topics": actual_topics
    }

    # 8. Build the final JSON structure matching the exact required format
    output_data = {
        "topics": topics,
        "docs": docs,
        "matches": matches,
        "metadata": metadata
    }

    # 9. Write JSON to file
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"JSON output successfully written to {output_json}")
        return True
    except Exception as e:
        print(f"Error writing JSON: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a PDF file and extract insights as JSON.')
    parser.add_argument('--input', required=True, help='Input PDF file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()
    
    print(f"Arguments: input={args.input}, output={args.output}")
    
    # Process the PDF and generate JSON
    success = process_pdf_to_json(args.input, args.output)
    if not success:
        print("Processing failed.")
        exit(1)
    else:
        print("Processing completed successfully.")

if __name__ == "__main__":
    main()
