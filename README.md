# DocExplore

## What is DocExplore?

DocExplore is an application that analyzes PDF documents to extract meaningful insights. It reads PDFs, identifies the main topics, groups similar content together, and presents an organized summary for easier understanding. Think of it as a factory line for document processing—your PDF enters one end and emerges as structured, organized information at the other.

## Setup Instructions

To set up DocExplore on your system, follow these steps:

1. **Prerequisites**
   - Python 3.11 or newer
   - Node.js and npm
   - Gramex framework

2. **Installation**
   ```bash
   # Create a virtual environment
   python -m venv gramex311
   
   # Activate the virtual environment
   # For Unix/MacOS:
   source gramex311/bin/activate
   # For Windows:
   gramex311\Scripts\activate
   
   # Install Python dependencies
   pip install gramex
   pip install sentence-transformers sklearn PyPDF2 numpy
   
   # Install Node.js dependencies
   npm install
   ```

3. **Running the Application**
   ```bash
   gramex
   ```
   The application will be available at http://localhost:9988

## Project Structure

Below is a comprehensive list of all files and directories in the DocExplore project:

| File/Directory | Description |
|----------------|-------------|
| **Backend Files** | |
| `filehandler.py` | Manages file uploads, processing, and progress tracking |
| `backend/` | Directory containing PDF processing scripts |
| `backend/extract_pdf_to_json.py` | Core script that extracts text and analyzes PDFs |
| `gramex.yaml` | Configuration file that handles URL routing |
| **Frontend Files** | |
| `upload.html` | Main page where users upload PDFs |
| `templates/` | Directory containing HTML templates |
| `templates/insights.tmpl.html` | Template for displaying analysis results |
| `style.css` | Styling for the application |
| `story.js` | JavaScript for frontend interactions |
| **Resource Directories** | |
| `uploads/` | Directory where uploaded PDFs and generated JSONs are stored |
| `node_modules/` | Directory containing Node.js dependencies |
| **Configuration Files** | |
| `package.json` | Node.js dependencies configuration |
| `package-lock.json` | Locked versions of Node.js dependencies |

## Predefined Variables

The application uses several predefined variables that control its behavior:

| Variable | Value | Location | Purpose |
|----------|-------|----------|---------|
| `MODEL_NAME` | "all-MiniLM-L6-v2" | `extract_pdf_to_json.py` | Defines which SentenceTransformer model to use for text analysis. This model offers a good balance between accuracy and performance. |
| `NUM_TOPICS` | 3 | `extract_pdf_to_json.py` | Sets the number of topic groups to identify in a document. Three topics provides a good balance between detail and simplicity for most documents. |
| `SIMILARITY_THRESHOLD` | 0.75 | `extract_pdf_to_json.py` | Minimum similarity score required to assign a paragraph to a topic. The higher the value, the more selective the matching. |
| `random_state` | 42 | `extract_pdf_to_json.py` | Seed value for the clustering algorithm to ensure consistent results across runs. |

## Backend Processing

This section explains how DocExplore processes PDFs, from upload to analysis to results.

### 1. File Upload

**What Happens**: The user uploads a PDF file through the web interface.

**How It Works**: The frontend captures the file, creates a session ID for tracking, and sends the PDF to the backend which validates and saves it.

**Files Involved**:
- `upload.html`: Contains the form for file uploads
- `story.js`: Handles the JavaScript for the upload process
- `filehandler.py`: Contains `PDFProcessHandler` which processes the upload
- `gramex.yaml`: Routes the upload request to the correct handler

**Functions Called**:

In `filehandler.py`, the `PDFProcessHandler` class handles the upload:

```python
# The PDFProcessHandler.post method receives the uploaded PDF
async def post(self):
    # Generate or get a session ID for progress tracking
    session_id = self.get_argument('session_id', str(uuid.uuid4()))
    # Log the start of processing
    logger.info(f"Starting PDF processing for session: {session_id}")
    
    # Update progress to 0% for upload stage
    self.update_progress(session_id, 'upload', 0)
    
    # Get the uploaded file from the request
    uploaded_files = self.request.files.get('file', [])
    if not uploaded_files:
        # Return an error if no file was uploaded
        self.set_status(400)
        self.write({"error": "No file uploaded"})
        return
    
    # Extract file information
    file_info = uploaded_files[0]
    file_name = file_info['filename']
    
    # Update progress to 50% for upload stage
    self.update_progress(session_id, 'upload', 50)
    
    # Validate that the file is a PDF
    if not file_name.lower().endswith('.pdf'):
        # Return an error if not a PDF
        self.set_status(400)
        self.write({"error": "File must be a PDF"})
        return
    
    # Prepare file paths
    base_name = os.path.splitext(file_name)[0]
    # Clean the filename for safe storage
    base_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '_', '-')]).strip()
    
    # Set paths for the input PDF and output JSON
    input_pdf_path = os.path.join('uploads', f"{base_name}.pdf")
    output_json_path = os.path.join('uploads', f"{base_name}.json")
    
    # Ensure uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    # Save the uploaded PDF
    with open(input_pdf_path, 'wb') as f:
        f.write(file_info['body'])
    
    # Update progress to 100% for upload stage
    self.update_progress(session_id, 'upload', 100)
    
    # Continue to extraction phase...
```

**Frontend Trigger**:

In `story.js`, the file upload is triggered by form submission:

```javascript
// Form submission event listener
form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get the selected file
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select a file to upload');
        return;
    }
    
    // Create form data with the file
    const formData = new FormData();
    formData.append('file', file);
    
    // Generate a session ID for progress tracking
    const sessionId = Math.random().toString(36).substring(2, 15);
    formData.append('session_id', sessionId);
    
    // Determine which endpoint to use based on file type
    let endpoint = file.name.toLowerCase().endsWith('.pdf') ? 
                  './process-pdf' : './process-json';
    
    // Send the file to the backend
    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
    .then(response => /* handle response */)
    .catch(error => /* handle error */);
});
```

**Flow Summary**:
1. User selects a PDF and submits the form
2. JavaScript creates a unique session ID and sends the file to `/process-pdf`
3. `PDFProcessHandler.post` receives the request and validates the file
4. The file is saved to the `uploads/` directory
5. Progress updates are sent at 0%, 50%, and 100% of the upload stage

### 2. PDF Processing

**What Happens**: After upload, the PDF is processed to extract text, identify topics, and generate a structured JSON output.

**How It Works**: The backend executes a Python script that reads the PDF, extracts text, converts text to numerical representations, clusters similar content, and saves the results.

**Files Involved**:
- `filehandler.py`: Contains code to execute the processing script
- `backend/extract_pdf_to_json.py`: The script that performs the actual processing
- `uploads/`: Directory where input and output files are stored

**Functions Called**:

In `filehandler.py`, after the upload is complete, processing begins:

```python
# Continuing from the upload process
# Start the extraction process
self.update_progress(session_id, 'extract', 0)

# Locate the Python interpreter and processing script
venv_python = os.path.join(os.getcwd(), 'gramex311', 'bin', 'python')
script_path = os.path.abspath('backend/extract_pdf_to_json.py')

# Update progress
self.update_progress(session_id, 'extract', 50)

# Prepare absolute paths for input and output
input_path = os.path.abspath(input_pdf_path)
output_path = os.path.abspath(output_json_path)

# Create command to run the processing script
command = [
    venv_python,
    script_path,
    '--input', input_path,
    '--output', output_path
]

# Start the analysis process
self.update_progress(session_id, 'analyze', 0)

# Execute the script as a subprocess
try:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=os.getcwd())
    )
    
    # Update progress
    self.update_progress(session_id, 'analyze', 100)
    self.update_progress(session_id, 'process', 0)
    
    # Verify the output file exists and contains valid JSON
    if not os.path.exists(output_json_path):
        raise Exception("Output file was not created")
    
    with open(output_json_path, 'r', encoding='utf-8') as f:
        json.load(f)  # Will raise JSONDecodeError if invalid
    
    # Complete the process
    self.update_progress(session_id, 'process', 100)
    self.update_progress(session_id, 'complete', 100)
    
    # Return success response
    self.write({
        "status": "success",
        "message": "PDF processed successfully",
        "outputFile": os.path.basename(output_json_path)
    })
    
except Exception as e:
    # Handle errors
    logger.error(f"Error processing PDF: {str(e)}")
    self.set_status(500)
    self.write({"error": f"Failed to process PDF: {str(e)}"})
```

In `extract_pdf_to_json.py`, the main processing occurs:

```python
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a PDF file and extract insights as JSON.')
    parser.add_argument('--input', required=True, help='Input PDF file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    args = parser.parse_args()
    
    # Process the PDF and generate JSON
    success = process_pdf_to_json(args.input, args.output)
    if not success:
        print("Processing failed.")
        exit(1)
    else:
        print("Processing completed successfully.")
```

**Flow Summary**:
1. `filehandler.py` prepares to run the extraction script
2. The script is executed as a subprocess with input and output paths
3. Progress updates are sent at key points (0%, 50%, 100%)
4. The script processes the PDF and generates a JSON file
5. The backend verifies the output and sends a success response

## Special Focus: PDF Analysis

The heart of DocExplore is the PDF analysis performed by `extract_pdf_to_json.py`. This section breaks down each function and how they work together.

### Text Extraction

The first step is extracting text from the PDF:

```python
def extract_text_from_pdf(pdf_path):
    """Extracts text from all pages of a PDF file."""
    try:
        # Open the PDF file using PyPDF2
        reader = PdfReader(pdf_path)
        print(f"Successfully opened PDF: {pdf_path}")
        print(f"Number of pages: {len(reader.pages)}")
    except Exception as e:
        # Handle any errors opening the file
        print(f"Error reading PDF: {e}")
        return ""
    
    text = ""
    # Process each page in the PDF
    for i, page in enumerate(reader.pages):
        try:
            # Extract text from the current page
            page_text = page.extract_text()
            if page_text:
                # Add the page text to our overall text
                text += page_text + "\n"
        except Exception as e:
            # Handle errors for individual pages
            print(f"Error extracting text from page {i}: {e}")
    
    # Return the combined text from all pages
    return text
```

### Paragraph Splitting

After extracting text, it's split into paragraphs:

```python
def split_into_paragraphs(text):
    """Splits text into paragraphs based on double newlines."""
    # Split text where there are two newlines (paragraph breaks)
    # Then remove any empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Split text into {len(paragraphs)} paragraphs")
    return paragraphs
```

### Computing Embeddings

This critical function converts text to numerical representations:

```python
def compute_embeddings(paragraphs, model):
    """Converts paragraphs to numerical embeddings using SentenceTransformer."""
    print(f"Computing embeddings for {len(paragraphs)} paragraphs")
    # Use the model to encode paragraphs into vectors
    # Each paragraph becomes a list of numbers representing its meaning
    embeddings = model.encode(paragraphs)
    print(f"Computed embeddings with shape: {embeddings.shape}")
    return embeddings
```

**What are Embeddings?**

Text embeddings are numerical representations of text that capture semantic meaning. Each paragraph is transformed into a vector (list of numbers) where similar text results in similar vectors. This allows the computer to mathematically compare text content.

### Clustering

Clustering groups similar paragraphs together:

```python
def perform_clustering(embeddings, num_clusters):
    """Groups similar paragraphs using K-means clustering."""
    # Determine how many samples we have
    n_samples = embeddings.shape[0]
    # Ensure we don't try to create more clusters than samples
    actual_clusters = min(num_clusters, n_samples) if n_samples > 0 else 1
    print(f"Performing clustering with {actual_clusters} clusters")
    
    # Create a K-means clustering model
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
    # Fit the model to our embeddings and get cluster assignments
    labels = kmeans.fit_predict(embeddings)
    # Get the center point of each cluster
    centroids = kmeans.cluster_centers_
    print(f"Clustering complete. Labels: {np.unique(labels, return_counts=True)}")
    
    return labels, centroids
```

**How K-means Clustering Works**:

1. The algorithm starts by placing a specified number of "centroid" points randomly
2. Each paragraph is assigned to the nearest centroid
3. Centroids are moved to the average position of all paragraphs assigned to them
4. Steps 2-3 are repeated until centroids stabilize
5. The result is groups of paragraphs with similar meanings

### Topic Keyword Extraction

After clustering, the system identifies a representative keyword for each topic:

```python
def extract_topic_keyword(paragraphs):
    """Identifies the most representative word for a group of paragraphs."""
    if not paragraphs:
        return "NoData"
    
    # Create a TF-IDF vectorizer to find important words
    # This excludes common English stopwords
    vectorizer = TfidfVectorizer(stop_words='english')
    # Create a matrix of word importance scores
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    # Get all the words the vectorizer found
    features = vectorizer.get_feature_names_out()
    # Calculate the total importance of each word
    scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    
    if scores.size > 0:
        # Find the index of the most important word
        top_index = int(np.argmax(scores))
        # Return that word as the topic name
        return features[top_index]
    else:
        # Default name if no important words found
        return "Topic"
```

**What is TF-IDF?**

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical measure that:
1. Gives higher values to words that appear frequently in a specific document
2. Gives lower values to words that appear in many documents
3. Helps identify distinctive words that characterize a topic

### Topic Assignment

This function determines which topics a paragraph belongs to:

```python
def assign_topics_to_doc(embedding, centroids, threshold):
    """Finds which topics a paragraph belongs to based on similarity."""
    similarities = []
    # Calculate similarity between the paragraph and each topic centroid
    for i, centroid in enumerate(centroids):
        # Cosine similarity calculation
        sim = np.dot(embedding, centroid) / (np.linalg.norm(embedding) * np.linalg.norm(centroid) + 1e-8)
        similarities.append(sim)
    
    # Return topic IDs and scores that meet the threshold
    return [(i, round(sim, 2)) for i, sim in enumerate(similarities) if sim >= threshold]
```

### JSON Output Generation

The main processing function ties everything together:

```python
def process_pdf_to_json(input_pdf, output_json):
    """Processes a PDF to extract insights and save as JSON."""
    # Extract text from PDF
    text = extract_text_from_pdf(input_pdf)
    if not text:
        print("No text extracted from PDF.")
        return False
    
    # Split text into paragraphs
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        print("No paragraphs found.")
        return False
    
    # Load model and compute embeddings
    try:
        model = SentenceTransformer(MODEL_NAME)
        embeddings = compute_embeddings(paragraphs, model)
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return False
    
    # Perform clustering
    try:
        labels, centroids = perform_clustering(embeddings, NUM_TOPICS)
        actual_topics = len(np.unique(labels))
    except Exception as e:
        print(f"Error during clustering: {e}")
        return False
    
    # Build topics array
    topics = []
    for i in range(actual_topics):
        # Find paragraphs in this cluster
        cluster_paragraphs = [paragraphs[j] for j in range(len(paragraphs)) if labels[j] == i]
        # Get representative keyword
        keyword = extract_topic_keyword(cluster_paragraphs)
        topics.append({
            "topic_id": i,
            "topic": keyword,
            "subtopic": f"Subtopic of {keyword}",
            "count": len(cluster_paragraphs)
        })
    
    # Build docs and matches arrays
    docs = []
    matches = []
    for j, para in enumerate(paragraphs):
        try:
            # Get paragraph embedding
            doc_embedding = embeddings[j]
            # Find which topics this paragraph belongs to
            assigned = assign_topics_to_doc(doc_embedding, centroids, SIMILARITY_THRESHOLD)
            # Get cluster label
            cluster_label = labels[j]
            # Calculate similarity score
            centroid = centroids[cluster_label]
            score = np.dot(doc_embedding, centroid) / (np.linalg.norm(doc_embedding) * np.linalg.norm(centroid) + 1e-8)
            score = round(score, 2)
            
            # Create document object
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
            
            # Create match objects
            for topic_id, sim in assigned:
                matches.append({
                    "doc": j,
                    "topic": topic_id,
                    "similarity": sim
                })
        except Exception as e:
            print(f"Error processing paragraph {j}: {e}")
            return False
    
    # Build metadata
    metadata = {
        "total_chunks": len(paragraphs),
        "embedding_dimension": int(embeddings.shape[1]),
        "total_topics": actual_topics
    }
    
    # Build final JSON structure
    output_data = {
        "topics": topics,
        "docs": docs,
        "matches": matches,
        "metadata": metadata
    }
    
    # Write JSON to file
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        print(f"JSON output successfully written to {output_json}")
        return True
    except Exception as e:
        print(f"Error writing JSON: {e}")
        return False
```

## Progress Tracking

DocExplore provides real-time progress updates during processing.

**How It Works**: The application uses Server-Sent Events (SSE) to send updates from the server to the browser without requiring page refreshes.

**Files Involved**:
- `filehandler.py`: Contains `ProgressHandler` and update functions
- `upload.html`: Contains frontend code to receive updates
- `story.js`: Establishes the SSE connection
- `gramex.yaml`: Routes progress requests

The `ProgressHandler` class in `filehandler.py` manages the SSE connection:

```python
class ProgressHandler(BaseHandler):
    """Handles Server-Sent Events for progress tracking."""
    
    async def get(self):
        """Streams progress updates to the client."""
        # Set SSE headers
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        self.set_header('Access-Control-Allow-Origin', '*')
        
        # Send initial connection event
        self.write("event: ping\n")
        self.write("data: {\"message\": \"Connection established\"}\n\n")
        await self.flush()
        
        # Get the session ID from the request
        session_id = self.get_argument('session', None)
        if not session_id:
            # Return error if no session ID provided
            self.write("event: error\n")
            self.write("data: {\"error\": \"No session ID provided\"}\n\n")
            await self.flush()
            return
        
        # Initialize progress for new sessions
        if session_id not in progress_updates:
            progress_updates[session_id] = {
                'step': 'upload',
                'progress': 0
            }
        
        # Keep track of last update to avoid duplicates
        last_update = None
        
        try:
            # Send updates as they become available
            for _ in range(1000):  # Limit to prevent infinite loops
                current_update = progress_updates.get(session_id)
                
                # Only send if there's a new update
                if current_update and current_update != last_update:
                    last_update = current_update.copy()
                    
                    # Send the update as an SSE event
                    self.write(f"event: progress\n")
                    self.write(f"data: {json.dumps(current_update)}\n\n")
                    await self.flush()
                    
                    # If process is complete, send final event and exit
                    if current_update.get('step') == 'complete':
                        await asyncio.sleep(0.5)
                        self.write("event: close\n")
                        self.write("data: {\"message\": \"Closing connection\"}\n\n")
                        await self.flush()
                        
                        # Clean up session data
                        if session_id in progress_updates:
                            del progress_updates[session_id]
                        break
                
                # Wait before checking again
                await asyncio.sleep(0.1)
        
        except Exception as e:
            # Handle any errors
            self.write("event: error\n")
            self.write(f"data: {{\"error\": \"Progress tracking failed: {str(e)}\"}}\n\n")
            await self.flush()
```

## Frontend Overview

The frontend of DocExplore consists of several key files that work together to provide the user interface.

### Upload Interface (`upload.html` and `story.js`)

The main interface allows users to:
- Upload PDF files via drag-and-drop or file selection
- See real-time progress during processing
- View error messages if problems occur
- Be redirected to results when processing completes

Key JavaScript functions in `story.js` include:
- File upload handling
- Progress bar updates
- Error display
- SSE connection management

### Results Interface (`templates/insights.tmpl.html`)

After processing, users are taken to the insights page, which:
- Loads the JSON file created during processing
- Displays topics identified in the document
- Shows paragraphs grouped by topic
- Provides interactive exploration options

## Routing Configuration

The `gramex.yaml` file controls how URLs are routed to handlers:

```yaml
url:
  # Main page route
  docexplore:
    pattern: /$YAMLURL/
    handler: FileHandler
    kwargs:
      path: upload.html
      template: true
      transform:
        "*.html": 
          function: template
  
  # Progress tracking endpoint
  progress:
    pattern: /$YAMLURL/progress
    handler: filehandler.ProgressHandler
  
  # PDF processing endpoint
  process-pdf:
    pattern: /$YAMLURL/process-pdf
    handler: filehandler.PDFProcessHandler
    kwargs:
      path: uploads/
  
  # JSON processing endpoint
  process-json:
    pattern: /$YAMLURL/process-json
    handler: filehandler.JSONProcessHandler
    kwargs:
      path: uploads/
  
  # Uploads access
  uploads:
    pattern: /$YAMLURL/uploads/(.*)
    handler: FileHandler
    kwargs:
      path: uploads/
      transform:
        "*.json":
          function: template
```

This configuration:
- Maps `/` to `upload.html` to display the upload form
- Maps `/progress` to `ProgressHandler` for SSE updates
- Maps `/process-pdf` to `PDFProcessHandler` for PDF uploads
- Maps `/uploads/...` to serve files from the uploads directory

## Error Handling

DocExplore implements comprehensive error handling at multiple levels:

1. **Frontend Validation**:
   - Checks if a file was selected
   - Verifies file type (PDF or JSON)
   - Displays user-friendly error messages

2. **Backend Validation**:
   - Validates uploaded files
   - Checks for required executables and scripts
   - Verifies output files exist and contain valid JSON

3. **Processing Error Handling**:
   - Catches and logs exceptions during PDF processing
   - Reports specific error messages for different failure points
   - Sends error status codes and messages to the frontend

4. **Progress Tracking Errors**:
   - Handles SSE connection issues
   - Reports errors through the event stream
   - Cleans up resources on error

## Notes

1. **Environment Assumptions**:
   - The application assumes Python is installed at `./gramex311/bin/python` on Unix/Mac or `.\gramex311\Scripts\python` on Windows
   - The `uploads/` directory is used for both input and output files
   - The application uses a fixed port (9988) by default

2. **Processing Details**:
   - The SentenceTransformer model "all-MiniLM-L6-v2" is a balance of accuracy and performance
   - The default of 3 topics works well for most documents but might not be optimal for very short or very long documents
   - Paragraphs are defined by double newlines, which may not match the document's actual structure perfectly

3. **Performance Considerations**:
   - Large PDFs may take longer to process
   - Text-based PDFs work best; scanned documents may not work well without OCR
   - The embedding process is computationally intensive and may require a capable machine

4. **Security Notes**:
   - The application sanitizes filenames but could use additional content validation
   - There are no user authentication mechanisms in the basic version
   - Consider adding file size limits for production use

---

DocExplore was developed by [Gramener](https://gramener.com/), a data science company that specializes in visual data analysis.
