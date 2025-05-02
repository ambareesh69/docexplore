# DocExplore: PDF Analysis and Visualization Tool

## What is DocExplore?

DocExplore is a specialized web application that processes PDF documents to extract, analyze, and visualize their content. Think of it as a document processing factory where your PDF travels along a conveyor belt, stopping at different stations that extract its text, identify main topics, group similar content, and finally present the organized information in an interactive format.

The system performs these key operations:
- Accepts PDF file uploads through a user-friendly interface
- Extracts and processes all textual content from the document
- Analyzes the text to identify main topics and related concepts using advanced AI techniques
- Groups similar content together using natural language processing
- Presents the organized information in a visual, interactive format

```
User --> [Upload PDF] --> [Extract Text] --> [Analyze Content] --> [Group Topics] --> [Visualize Results]
```

## Setup Instructions

To set up DocExplore on your system, follow these steps:

### 1. System Requirements
   - Python 3.11 or newer
   - Node.js and npm
   - Gramex framework (a data visualization server)

### 2. Installation Process
   ```bash
   # Create a virtual environment
   python -m venv gramex311
   
   # Activate the virtual environment
   # For Unix/MacOS:
   source gramex311/bin/activate
   # For Windows:
   gramex311\Scripts\activate
   
   # Install required Python packages
   pip install gramex
   pip install sentence-transformers scikit-learn PyPDF2 numpy
   
   # Install frontend dependencies
   npm install
   ```

### 3. Starting the Application
   ```bash
   gramex
   ```
   
   Once started, access DocExplore by navigating to http://localhost:9988 in your web browser.

## Project Structure

The DocExplore project consists of the following files and directories:

### Core Files
- **filehandler.py**: Manages file uploads, processing workflows, and progress tracking
- **backend/extract_pdf_to_json.py**: Performs PDF text extraction and topic analysis
- **gramex.yaml**: Configuration file that defines routing and request handling
- **upload.html**: Main interface for uploading PDF files
- **style.css**: Defines the visual styling of the application
- **story.js**: Handles frontend interactions and communicates with backend
- **templates/insights.tmpl.html**: Template for displaying analysis results

### Support Directories
- **uploads/**: Storage location for uploaded PDFs and generated JSON files
- **node_modules/**: Contains frontend JavaScript dependencies
- **backend/**: Houses backend processing scripts
- **templates/**: Contains HTML templates used by the application
- **gramex311/**: Virtual environment containing Python dependencies

### Configuration Files
- **package.json**: Lists npm dependencies and project information
- **package-lock.json**: Detailed dependency information for npm
- **.gitignore**: Specifies files to exclude from version control
- **.gitattributes**: Defines attributes for specific paths
- **.gitlab-ci.yml**: Configuration for GitLab CI/CD pipeline
- **config.yaml**: Additional configuration for Gramex

### Other Files
- **default.tmpl.html**: Default template for HTML pages
- **emailauth.template.html**: Template for email authentication
- **.DS_Store**: macOS directory attributes file (should be ignored)

## Predefined Variables

The application uses several predefined variables that control its behavior:

| Variable | Value | Location | Purpose |
|----------|-------|----------|---------|
| MODEL_NAME | "all-MiniLM-L6-v2" | extract_pdf_to_json.py | Specifies which SentenceTransformer model to use for text analysis - this model provides a good balance of speed and accuracy for semantic text processing while requiring less computational resources |
| NUM_TOPICS | 3 | extract_pdf_to_json.py | Defines the default number of topic groups to identify in a document - this number represents a balance between providing enough detail without overwhelming users with too many categories |
| SIMILARITY_THRESHOLD | 0.75 | extract_pdf_to_json.py | Minimum similarity score (0-1) required to assign a paragraph to a topic - higher values create more distinct topic groups with less overlap, reducing false categorizations |
| random_state | 42 | extract_pdf_to_json.py | Ensures consistent clustering results between runs - this fixed seed makes the analysis reproducible for the same input document |
| UPLOADS_DIR | "uploads/" | filehandler.py | Directory where uploaded files and output JSONs are stored - this centralizes file storage for easier management |
| DEFAULT_RANGE_VALUE | 0.50 | insights.tmpl.html | Default similarity threshold in the UI slider - represents a balanced starting point for visualizations |
| MIN_RANGE_VALUE | 0.40 | insights.tmpl.html | Minimum similarity threshold allowed in the UI - prevents showing extremely loose associations |
| MAX_RANGE_VALUE | 0.95 | insights.tmpl.html | Maximum similarity threshold allowed in the UI - prevents filtering out nearly all results |
| charsPerPixel | 20 | insights.tmpl.html | Controls visualization sizing - determines how many characters correspond to one pixel width in the document visualization |
| progressInterval | 0.1 | filehandler.py | Controls how frequently progress updates are sent (seconds) - balances responsiveness with server load |
| PYTHONPATH | os.getcwd() | filehandler.py | Ensures Python can find modules in the current directory - needed when running extract_pdf_to_json.py as a subprocess |

## Backend Processing Workflow

### 1. PDF File Upload

**What Happens**: You upload a PDF file through the web interface.

**How It Works**: Imagine your PDF arriving at the first station of our document processing factory. Here, the system receives your PDF file, validates it to ensure it's actually a PDF, and stores it in the uploads directory for processing.

**Files Involved**:
- `upload.html`: Provides the interface for file selection
- `story.js`: Handles file upload and progress tracking
- `filehandler.py`: Processes the uploaded file
- `gramex.yaml`: Routes the upload request to the appropriate handler

**Processing Flow**:

1. **Frontend Initiation**:
   The upload process begins when you submit the form in `upload.html`:

   ```javascript
   // In story.js (via upload.html)
   // When form is submitted
   form.addEventListener('submit', function(e) {
     e.preventDefault();  // Prevent default form submission
     
     const file = fileInput.files[0];  // Get the selected file
     if (!file) {
       showError('Please select a file to upload');
       return;
     }
     
     // Create form data to send to server
     const formData = new FormData();
     formData.append('file', file);
     
     // Generate a unique session ID for tracking progress
     const sessionId = Math.random().toString(36).substring(2, 15);
     formData.append('session_id', sessionId);
     
     // Determine endpoint based on file type
     let endpoint = '';
     if (file.name.toLowerCase().endsWith('.pdf')) {
       endpoint = './process-pdf';
     } else {
       showError('File must be a PDF');
       return;
     }
     
     // Create EventSource for progress updates
     const progressSource = new EventSource(`./progress?session=${sessionId}`);
     
     // Send the file to the server
     fetch(endpoint, {
       method: 'POST',
       body: formData
     })
     // ... response handling follows
   });
   ```

2. **Backend Reception**:
   The `PDFProcessHandler` in `filehandler.py` receives and processes the file:

   ```python
   # In filehandler.py - The PDF handling function that receives uploads
   async def post(self):
       # Get the session ID from the request or generate a new one
       session_id = self.get_argument('session_id', str(uuid.uuid4()))
       logger.info(f"Starting PDF processing for session: {session_id}")
       
       # Send initial progress update
       self.update_progress(session_id, 'upload', 0)
       await asyncio.sleep(0.1)
       
       # Get the uploaded file from the request
       uploaded_files = self.request.files.get('file', [])
       if not uploaded_files:
           # Handle case where no file was uploaded
           self.set_status(400)
           self.write({"error": "No file uploaded"})
           return
       
       # Extract file information
       file_info = uploaded_files[0]
       file_name = file_info['filename']
       
       # Update progress to indicate file received
       self.update_progress(session_id, 'upload', 50)
       await asyncio.sleep(0.1)
       
       # Validate that the file is a PDF
       if not file_name.lower().endswith('.pdf'):
           self.set_status(400)
           self.write({"error": "File must be a PDF"})
           return
       
       # Prepare file paths for saving
       base_name = os.path.splitext(file_name)[0]
       # Clean filename to ensure it's valid for the filesystem
       base_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '_', '-')]).strip()
       
       # Define paths for input and output files
       input_pdf_path = os.path.join('uploads', f"{base_name}.pdf")
       output_json_path = os.path.join('uploads', f"{base_name}.json")
       
       # Ensure uploads directory exists
       os.makedirs('uploads', exist_ok=True)
       
       # Save the uploaded PDF file
       with open(input_pdf_path, 'wb') as f:
           f.write(file_info['body'])
       
       # Update progress to indicate file saved
       self.update_progress(session_id, 'upload', 100)
       await asyncio.sleep(0.1)
       
       # Continue to PDF processing stage
       # ... processing code follows
   ```

**Flow Summary**:
1. User selects a PDF file in the browser
2. JavaScript creates FormData with the file and a unique session ID
3. File is sent to the `/process-pdf` endpoint via fetch
4. Backend validates the file is a PDF
5. Backend saves the file to the `uploads/` directory
6. Progress updates are sent at 0%, 50%, and 100% of the upload stage

```
                                 +---------- 0% progress ------------+
                                 |                                   |
User --> [Browser: upload.html] -+-> [Server: filehandler.py] --> [Save PDF]
                 |                                                   |
                 v                                                   v
               [SSE]                                           100% progress  
                 ^                                                   
                 |                                                   
          [Progress Updates] <------------------------------------+     
```

### 2. PDF Text Extraction and Analysis

**What Happens**: After saving your PDF, the system extracts and analyzes its textual content.

**How It Works**: In our document processing factory, your PDF now moves to the analysis station. Here, specialized machinery (Python code) opens your PDF, extracts all the text from each page, and processes it to identify the main topics and organize related content together.

**Files Involved**:
- `filehandler.py`: Initiates the processing workflow
- `backend/extract_pdf_to_json.py`: Performs the actual text extraction and analysis

**Processing Flow**:

1. **Initiating Text Extraction**:
   After the upload completes, `filehandler.py` continues with text extraction by launching a subprocess:

   ```python
   # Begin text extraction phase
   self.update_progress(session_id, 'extract', 0)
   await asyncio.sleep(0.1)
   
   # Locate the Python interpreter and processing script
   venv_python = os.path.join(os.getcwd(), 'gramex311', 'bin', 'python')
   if not os.path.exists(venv_python):
       raise Exception(f"Virtual environment Python not found at {venv_python}")
   
   script_path = os.path.abspath('backend/extract_pdf_to_json.py')
   if not os.path.exists(script_path):
       raise Exception(f"Script not found at {script_path}")
   
   # Update progress to indicate preparation complete
   self.update_progress(session_id, 'extract', 50)
   await asyncio.sleep(0.1)
   
   # Prepare absolute paths for processing
   input_path = os.path.abspath(input_pdf_path)
   output_path = os.path.abspath(output_json_path)
   
   # Create the command to run the extraction script
   command = [
       venv_python,  # Python interpreter
       script_path,  # Path to the processing script
       '--input', input_path,  # Input PDF file
       '--output', output_path  # Output JSON file
   ]
   
   # Begin analysis phase
   self.update_progress(session_id, 'analyze', 0)
   await asyncio.sleep(0.1)
   
   # Execute the extraction script as a subprocess
   try:
       result = subprocess.run(
           command,
           capture_output=True,  # Capture stdout and stderr
           text=True,  # Return strings instead of bytes
           check=True,  # Raise exception on non-zero exit
           env=dict(os.environ, PYTHONPATH=os.getcwd())  # Set environment variables
       )
   ```

2. **Main Script Entry Point**:
   The `extract_pdf_to_json.py` script starts with the `main()` function that parses arguments and calls the main processing function:

   ```python
   # In extract_pdf_to_json.py - Main entry point
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
   ```

3. **PDF Processing Workflow**:
   The main function `process_pdf_to_json` orchestrates the entire analysis process:

   ```python
   # In extract_pdf_to_json.py - The main processing function
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
   
       # 5. Build the "topics" array
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
   
       # 6. Build the "docs" and "matches" arrays
       docs = []
       matches = []
       # ... building docs and matches
   
       # 7. Write JSON to file
       try:
           with open(output_json, "w", encoding="utf-8") as f:
               json.dump(output_data, f, indent=4)
           print(f"JSON output successfully written to {output_json}")
           return True
       except Exception as e:
           print(f"Error writing JSON: {e}")
           return False
   ```

4. **PDF Text Extraction**:
   The `extract_text_from_pdf` function handles the actual PDF content extraction:

   ```python
   # In extract_pdf_to_json.py - Extract text from PDF
   def extract_text_from_pdf(pdf_path):
       """Extract text from all pages of a PDF document."""
       try:
           # Open the PDF file using PyPDF2
           reader = PdfReader(pdf_path)
           print(f"Successfully opened PDF: {pdf_path}")
           print(f"Number of pages: {len(reader.pages)}")
       except Exception as e:
           # Handle any errors in opening the file
           print(f"Error reading PDF: {e}")
           return ""
       
       # Initialize empty text string
       text = ""
       # Process each page in the PDF
       for i, page in enumerate(reader.pages):
           try:
               # Extract text from the current page
               page_text = page.extract_text()
               if page_text:
                   # Add the page text to our collected text
                   text += page_text + "\n"
           except Exception as e:
               # Handle errors in extracting text from this page
               print(f"Error extracting text from page {i}: {e}")
       return text  # Return the combined text from all pages
   ```

5. **Text Processing**:
   After extraction, the text is split into paragraphs for further processing:

   ```python
   # In extract_pdf_to_json.py - Split text into paragraphs
   def split_into_paragraphs(text):
       """Split text into paragraphs based on double newlines."""
       # Split the text wherever there are two newlines (paragraph breaks)
       paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
       print(f"Split text into {len(paragraphs)} paragraphs")
       return paragraphs
   ```

6. **Computing Embeddings**:
   Text paragraphs are converted to numerical representations (embeddings) using SentenceTransformer:

   ```python
   # In extract_pdf_to_json.py - Convert text to embeddings
   def compute_embeddings(paragraphs, model):
       """Convert paragraphs to numerical embeddings using SentenceTransformer."""
       print(f"Computing embeddings for {len(paragraphs)} paragraphs")
       # Transform text into numerical vectors
       embeddings = model.encode(paragraphs)
       print(f"Computed embeddings with shape: {embeddings.shape}")
       return embeddings
   ```

7. **Topic Identification (Clustering)**:
   The embeddings are grouped into topics using K-means clustering:

   ```python
   # In extract_pdf_to_json.py - Cluster similar paragraphs
   def perform_clustering(embeddings, num_clusters):
       """Group similar paragraphs using K-means clustering."""
       # Determine how many actual clusters to create
       n_samples = embeddings.shape[0]
       actual_clusters = min(num_clusters, n_samples) if n_samples > 0 else 1
       print(f"Performing clustering with {actual_clusters} clusters")
       
       # Apply K-means clustering to group similar paragraphs
       kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
       labels = kmeans.fit_predict(embeddings)
       centroids = kmeans.cluster_centers_
       
       print(f"Clustering complete. Labels: {np.unique(labels, return_counts=True)}")
       return labels, centroids
   ```

8. **Topic Naming**:
   The system identifies representative keywords for each topic group:

   ```python
   # In extract_pdf_to_json.py - Find representative keywords
   def extract_topic_keyword(paragraphs):
       """Identify the most representative keyword for a group of paragraphs."""
       if not paragraphs:
           return "NoData"
       
       # Use TF-IDF to find important words
       vectorizer = TfidfVectorizer(stop_words='english')
       tfidf_matrix = vectorizer.fit_transform(paragraphs)
       features = vectorizer.get_feature_names_out()
       
       # Calculate importance scores for each word
       scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
       
       if scores.size > 0:
           # Find the most important word
           top_index = int(np.argmax(scores))
           return features[top_index]
       else:
           return "Topic"
   ```

9. **Assigning Topics to Documents**:
   Each paragraph is matched to the most similar topics:

   ```python
   # In extract_pdf_to_json.py - Match paragraphs to topics
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
   ```

10. **Processing Completion**:
    After the script completes, `filehandler.py` verifies the results and sends a success response:

    ```python
    # Back in filehandler.py - After subprocess completes
    # Complete analyze phase
    self.update_progress(session_id, 'analyze', 100)
    await asyncio.sleep(0.1)
    
    # Begin process phase
    self.update_progress(session_id, 'process', 0)
    await asyncio.sleep(0.1)
    
    # Verify output file exists
    if not os.path.exists(output_json_path):
        raise Exception("Output file was not created")
    
    # Verify valid JSON
    with open(output_json_path, 'r', encoding='utf-8') as f:
        json.load(f)  # Will raise JSONDecodeError if invalid
    
    # Complete process phase
    self.update_progress(session_id, 'process', 100)
    await asyncio.sleep(0.1)
    
    # Mark processing as complete
    self.update_progress(session_id, 'complete', 100)
    await asyncio.sleep(0.1)
    
    # Send success response
    self.write({
        "status": "success",
        "message": "PDF processed successfully",
        "outputFile": os.path.basename(output_json_path)
    })
    ```

**Flow Summary**:
1. Backend initiates `extract_pdf_to_json.py` as a subprocess
2. The script extracts text from the PDF using PyPDF2
3. Text is split into paragraphs
4. SentenceTransformer converts paragraphs to numerical embeddings
5. K-means clustering groups similar paragraphs
6. TF-IDF identifies representative keywords for each topic
7. Similarities between paragraphs and topics are calculated
8. Results are compiled into structured JSON
9. Backend verifies output and sends success response

```
PDF File ---> [extract_text_from_pdf] ---> Text
  |
  v
  Text ---> [split_into_paragraphs] ---> Paragraphs
  |
  v
Paragraphs ---> [compute_embeddings] ---> Embeddings
  |
  v
Embeddings ---> [perform_clustering] ---> Topic Groups
  |
  v
Topic Groups ---> [extract_topic_keyword] ---> Named Topics
  |
  v
Named Topics + Paragraphs ---> [assign_topics_to_doc] ---> Matched Content
  |
  v
JSON Output
```

### 3. Progress Tracking System

**What Happens**: Throughout processing, you receive real-time updates about the status.

**How It Works**: In our document processing factory, a status board displays the current stage and progress of your document. The system uses Server-Sent Events (SSE) to stream progress updates from the server to your browser in real-time.

**Files Involved**:
- `filehandler.py`: Manages the progress tracking and updates via the `ProgressHandler` class
- `story.js`: Establishes the connection and handles updates
- `upload.html`: Displays the progress to the user

**Processing Flow**:

1. **Frontend Connection Setup**:
   When you upload a file, the JavaScript in `upload.html` establishes an SSE connection:

   ```javascript
   // Create a connection for progress updates
   const progressSource = new EventSource(`./progress?session=${sessionId}`);
   
   // Set up event listeners for different types of events
   progressSource.addEventListener('open', function(event) {
     console.log('SSE connection opened');
     updateProgress('upload', 0);
     processingMessage.textContent = 'Upload started...';
   });
   
   progressSource.addEventListener('progress', function(event) {
     console.log('Progress event received:', event.data);
     try {
       const data = JSON.parse(event.data);
       
       // Update the UI based on progress
       updateProgress(data.step, data.progress);
       
       // Update status message based on current step
       switch(data.step) {
         case 'upload':
           processingMessage.textContent = `Uploading file... (${Math.round(data.progress)}%)`;
           break;
         case 'extract':
           processingMessage.textContent = `Extracting text from PDF... (${Math.round(data.progress)}%)`;
           break;
         case 'analyze':
           processingMessage.textContent = `Analyzing document content... (${Math.round(data.progress)}%)`;
           break;
         case 'process':
           processingMessage.textContent = `Processing document insights... (${Math.round(data.progress)}%)`;
           break;
         case 'complete':
           processingMessage.textContent = 'Process complete! Redirecting to insights...';
           break;
       }
     } catch (error) {
       console.error('Error parsing progress data:', error);
       processingMessage.textContent = 'Processing file...';
     }
   });
   ```

2. **Backend Progress Handler**:
   The `ProgressHandler` in `filehandler.py` manages the SSE connection:

   ```python
   class ProgressHandler(BaseHandler):
       """Handle Server-Sent Events for progress updates."""
       
       async def get(self):
           """Stream progress updates to the client."""
           # Set headers for SSE connection
           self.set_header('Content-Type', 'text/event-stream')
           self.set_header('Cache-Control', 'no-cache')
           self.set_header('Connection', 'keep-alive')
           self.set_header('Access-Control-Allow-Origin', '*')
           self.set_header('X-Accel-Buffering', 'no')  # Disable nginx buffering
           
           # Send initial connection event
           self.write("event: ping\n")
           self.write("data: {\"message\": \"Connection established\"}\n\n")
           await self.flush()
           
           # Get session ID from request
           session_id = self.get_argument('session', None)
           if not session_id:
               self.write("event: error\n")
               self.write("data: {\"error\": \"No session ID provided\"}\n\n")
               await self.flush()
               return
           
           # Initialize progress tracker if needed
           if session_id not in progress_updates:
               progress_updates[session_id] = {
                   'step': 'upload',
                   'progress': 0
               }
           
           # Track last update to avoid duplicates
           last_update = None
           
           try:
               # Send updates in a loop
               for _ in range(1000):  # Limit to prevent infinite loops
                   current_update = progress_updates.get(session_id)
                   
                   # Only send if there's a new update
                   if current_update and current_update != last_update:
                       last_update = current_update.copy()
                       
                       # Format and send the update
                       self.write(f"event: progress\n")
                       self.write(f"data: {json.dumps(current_update)}\n\n")
                       await self.flush()
                       
                       logger.info(f"Sent progress update to client: {current_update}")
                       
                       # If process is complete, send closing event
                       if current_update.get('step') == 'complete':
                           await asyncio.sleep(0.5)  # Give time for the client to receive the complete event
                           self.write("event: close\n")
                           self.write("data: {\"message\": \"Closing connection\"}\n\n")
                           await self.flush()
                           
                           # Clean up session data
                           if session_id in progress_updates:
                               del progress_updates[session_id]
                           break
                   
                   # Wait briefly before checking again
                   await asyncio.sleep(0.1)
                   
           except Exception as e:
               # Handle connection errors
               logger.error(f"Error in progress handler: {str(e)}")
               self.write("event: error\n")
               self.write(f"data: {{\"error\": \"Progress tracking failed: {str(e)}\"}}\n\n")
               await self.flush()
   ```

3. **Progress Updates**:
   The `PDFProcessHandler` updates progress at key points throughout processing:

   ```python
   def update_progress(self, session_id, step, progress=0):
       """Update progress for a given session."""
       if not session_id:
           return
           
       # Store the update in the global dictionary
       progress_updates[session_id] = {
           'step': step,  # Current processing step
           'progress': progress  # Percentage complete (0-100)
       }
       logger.debug(f"Progress updated for session {session_id}: {step} - {progress}%")
   ```

4. **Progress Visualization**:
   The frontend updates the visual elements to reflect progress:

   ```javascript
   // In upload.html - Update progress indicators
   function updateProgress(step, progress) {
     // Calculate which step index we're at
     const steps = ['upload', 'extract', 'analyze', 'process', 'complete'];
     const stepIndex = steps.indexOf(step);
     
     // Calculate overall progress across all steps
     const totalSteps = steps.length;
     const stepSize = 100 / totalSteps;
     const totalProgress = stepIndex * stepSize + (progress / 100) * stepSize;
     
     // Update progress bar
     progressBar.style.width = `${totalProgress}%`;
     progressBar.setAttribute('aria-valuenow', totalProgress);
     
     // Update step indicators
     document.querySelectorAll('.progress-step').forEach((el, index) => {
       if (index < stepIndex) {
         el.classList.remove('active');
         el.classList.add('completed');
       } else if (index === stepIndex) {
         el.classList.add('active');
         el.classList.remove('completed');
       } else {
         el.classList.remove('active', 'completed');
       }
     });
   }
   ```

**Flow Summary**:
1. Frontend establishes SSE connection with session ID
2. Backend sends initial "connection established" message
3. As processing occurs, backend calls update_progress() at key points
4. ProgressHandler continuously checks for new updates
5. When updates occur, they're sent to the frontend
6. Frontend updates the UI to reflect current status
7. When processing completes, connection is closed and session data cleaned up

```
  Backend Processing                  SSE Connection                Frontend Display
  ------------------                  -------------                ----------------
[Update Progress 0%] -----> [ProgressHandler] -----> [EventSource] -----> [Progress Bar: 0%]
         |                         |                       |                    |
         v                         v                       v                    v
[Update Progress 50%] ----> [Send SSE Event] -----> [progress Event] ---> [Progress Bar: 50%]
         |                         |                       |                    |
         v                         v                       v                    v
[Update Progress 100%] ---> [Send SSE Event] -----> [progress Event] ---> [Progress Bar: 100%]
         |                         |                       |                    |
         v                         v                       v                    v
  [Mark Complete] --------> [Send close Event] -----> [close Event] ------> [Redirect]
```

### 4. Result Presentation

**What Happens**: After processing completes, you're redirected to view the analysis results in an interactive visualization.

**How It Works**: In our document processing factory, your processed document now moves to the display gallery. The system loads the generated JSON file and renders it as an interactive visualization that allows you to explore topics, connections, and document structure.

**Files Involved**:
- `story.js`: Handles the redirection and visualization
- `templates/insights.tmpl.html`: Renders the visualization interface
- `gramex.yaml`: Routes the request to the insights template

**Processing Flow**:

1. **Frontend Redirection**:
   When processing completes, JavaScript in `upload.html` redirects to the insights page:

   ```javascript
   // In upload.html - After successful processing
   .then(data => {
     console.log('Upload successful:', data);
     updateProgress('complete', 100);
     
     // Allow a brief moment for the final update
     setTimeout(() => {
       // Close the progress connection
       progressSource.close();
       // Redirect to insights page with the filename
       window.location.href = './insights?json=' + data.outputFile;
     }, 100);
   })
   ```

2. **Insights Template Rendering**:
   The `insights.tmpl.html` template loads and creates the visualization framework:

   ```html
   <!-- In insights.tmpl.html -->
   {% set json_file = handler.get_argument('json', '') %}
   {% set title = 'Document Insights' %}
   
   <!-- ... header code ... -->
   
   {% if not json_file %}
     <div class="alert alert-danger text-center h4 fw-normal">No file specified.</div>
     <p class="text-center my-5">
       <a class="btn btn-primary" href="/">Go to upload page</a>
     </p>
   {% else %}
     <div class="col-12 col-sm-9 col-md-6 col-lg-4 mx-auto">
       <p>Explore your document insights by scrolling through the story below.</p>
     </div>
     <section id="scrolly" class="row position-relative py-3">
       <article class="steps col-6 col-lg-4"></article>
       <figure class="col-6 col-lg-8 position-sticky figure-panel">
         <div id="panel-topics" data-panel="topics"></div>
         <div id="panel-text" data-panel="text"></div>
         <div id="panel-network" data-panel="network">
           <svg id="doc-network" class="w-100 vh-100" viewBox="0 0 600 600"></svg>
         </div>
       </figure>
     </section>
     
     <!-- ... footer and modal code ... -->
     
     <script type="module">
       window.dataLink = "/uploads/{{ json_file }}";
       window.charsPerPixel = 20;
     </script>
     <script type="module" src="../story.js"></script>
   {% end %}
   ```

3. **JSON Data Loading**:
   The `story.js` script loads the JSON data and initializes the visualization:

   ```javascript
   // In story.js - Loading data and initializing visualizations
   const data = await fetch(dataLink).then((r) => r.json());
   
   data.topics.forEach((topic, i) => (topic.index = i));
   data.docs.forEach((doc, i) => (doc.index = i));
   data.topicColor = d3.scaleOrdinal(d3.schemeCategory10).domain(data.topics.map((d) => d.index));
   
   // Flatten the column hierarchies into linear hierarchies via sequence().
   data.topicHierarchy = sequence(data.topics, ["topic", "subtopic"]);
   data.docHierarchy = sequence(data.docs, ["chapter", "section", "para"]);
   ```

4. **Interactive Story Generation**:
   The script then creates an interactive story to guide the user through the document insights:

   ```javascript
   // In story.js - Creating the interactive story
   function drawStory() {
     // Initialize topic visualization
     const tree = drawTopics(data);
     
     // Group topics and subtopics
     const subtopics = {};
     let lastTopic;
     for (const row of tree.tree) {
       if (row[LEVEL] == 1) subtopics[(lastTopic = row[GROUP])] = [];
       if (row[LEVEL] == 2) subtopics[lastTopic].push(row);
     }
     
     const topics = Object.keys(subtopics);
     const chapters = new Set(data.docs.map((doc) => doc.chapter));
     const sections = new Set(data.docs.map((doc) => doc.section));
     
     // Create the story steps
     const docStory = html`
       <div class="step p-4" data-panel-target="topics" data-group="">
         <h2>Here are the topics most discussed in the document.</h2>
       </div>
       ${Object.entries(subtopics).map(
         ([topic, subtopics], i) =>
           html`<div class="step p-4" data-panel-target="topics" data-group="${topic}">
             <h2>
               ${i == 0
                 ? html`<span style="color:${data.topicColor(topic)}">${topic}</span> was the most discussed`
                 : html`... followed by <span style="color:${data.topicColor(topic)}">${topic}</span>`}
             </h2>
           </div>`,
       )}
       // ... additional story steps
     `;
     
     // Render the story
     render(
       [
         docStory,
         networkStory,
         // ... additional stories
       ],
       document.querySelector(".steps"),
     );
     
     // Initialize scrollama for scroll-based interaction
     if (scroller) scroller.destroy();
     scroller = scrollama()
       .setup({ step: ".step" })
       .onStepEnter(drawPanel)
       .onStepExit(({ element }) => element.classList.remove("text-bg-success"));
   }
   ```

5. **Visualization Rendering**:
   The script renders various visualizations as the user scrolls through the story:

   ```javascript
   // In story.js - Topic tree visualization
   function drawTopics({ topics, docs, matches, topicColor, group }) {
     // Apply minimum similarity filter from slider
     const minSimilarity = document.querySelector("#min-similarity").value;
     
     // Update topic counts based on matches
     topics.forEach((topic) => (topic.count = 0));
     docs.forEach((doc) => {
       doc.count = 0;
       doc.topics = [];
     });
     matches
       .filter(({ similarity }) => similarity > minSimilarity)
       .forEach(({ doc, topic }) => {
         topics[topic].count++;
         docs[doc].count++;
         docs[doc].topics.push(topic);
       });
     
     // Render the insight tree
     const tree = insightTree("#panel-topics", {
       data: topics,
       groups: ["topic", "subtopic"],
       metrics: ["count"],
       sort: "-count",
       impact: "-count",
       totalGroup: "All Topics",
       render: (el, { tree }) => {
         // Render the tree using lit-html
         // ... rendering code
       },
     });
     
     // Update the tree view
     tree.update({ level: 1 });
     if (group) tree.show((d) => d[LEVEL] == 0 || d[GROUP] == group || d[PARENT]?.[GROUP] == group);
     return tree;
   }
   
   // Document map visualization
   function drawText({ filter }) {
     // Apply minimum similarity filter
     const minSimilarity = $minSimilarity.value;
     data.docTopicMap = data.matches
       .filter(({ similarity }) => similarity >= minSimilarity)
       .map(({ doc, topic }) => [docMap[doc], topicMap[topic]]);
     
     // Create the document map visualization
     documapChart = documap("#panel-text", {
       topics: data.topicHierarchy,
       docs: data.docHierarchy,
       docTopicMap: data.docTopicMap,
       topicLabel: (d) => d.name,
       markerStyle: (toggle) =>
         toggle.attr("r", 3).style("fill", ([, topicId]) => data.topicColor(data.topicHierarchy[topicId].topic)),
       d3,
     });
     
     // Style and update the visualization elements
     // ... styling code
     
     // Apply filters from story navigation
     documapChart.update({ topics: (d) => filter.topics?.includes(d.topic) });
   }
   
   // Network visualization
   function drawNetwork({ topics, topicColor }) {
     // Create network visualization showing topic relationships
     // ... network visualization code
   }
   ```

**Flow Summary**:
1. After processing completes, user is redirected to `/insights?json=filename.json`
2. The server loads `insights.tmpl.html` template
3. The template creates a framework for the visualization
4. `story.js` loads the JSON data via fetch
5. Various visualizations are initialized (topic tree, document map, network)
6. An interactive story is created to guide the user through insights
7. Scrollama library enables scroll-based interaction with visualizations
8. User can explore topics, documents, and relationships interactively

```
Processing Complete --> [Browser Redirect] --> [insights.tmpl.html]
       |                                               |
       v                                               v
JSON File in uploads/ <------------------------ [story.js: fetch]
       |                                               |
       v                                               v
   [JSON Data] --------------------------------> [Visualization]
                                                       |
                                                       v
                                                 [User Interaction]
                                                       |
                                                       v
                                                [Document Insights]
```

## Routing Configuration

DocExplore uses the `gramex.yaml` file to define how requests are routed to different handlers. This acts like a traffic controller directing each request to the appropriate part of the application.

```yaml
# In gramex.yaml - URL routing configuration
url:
  docexplore:
    pattern: /$YAMLURL/
    handler: FileHandler
    kwargs:
      path: upload.html
      template: true
      transform:
        "*.html": 
          function: template
  
  insights:
    pattern: /$YAMLURL/insights
    handler: FileHandler
    kwargs:
      path: templates/insights.tmpl.html
      template: true
  
  progress:
    pattern: /$YAMLURL/progress
    handler: filehandler.ProgressHandler
  
  process-pdf:
    pattern: /$YAMLURL/process-pdf
    handler: filehandler.PDFProcessHandler
    kwargs:
      path: uploads/
  
  process-json:
    pattern: /$YAMLURL/process-json
    handler: filehandler.JSONProcessHandler
    kwargs:
      path: uploads/
  
  uploads:
    pattern: /$YAMLURL/uploads/(.*)
    handler: FileHandler
    kwargs:
      path: uploads/
      transform:
        "*.json":
          function: template

  node-modules:
    pattern: /$YAMLURL/node_modules/(.*)
    handler: FileHandler
    kwargs:
      path: node_modules/

  assets:
    pattern: /$YAMLURL/(.+\.(js|css|png|jpg|gif|svg|eot|ttf|woff|woff2))
    handler: FileHandler
    kwargs:
      path: .
```

This configuration maps URLs to specific handlers:

- `/` → Shows the upload page via `upload.html`
- `/insights` → Displays analysis results via `insights.tmpl.html`
- `/progress` → Streams progress updates via `ProgressHandler`
- `/process-pdf` → Handles PDF uploads via `PDFProcessHandler`
- `/process-json` → Handles JSON uploads via `JSONProcessHandler`
- `/uploads/...` → Serves files from the uploads directory
- `/node_modules/...` → Serves frontend dependencies
- `/*.js`, `/*.css`, etc. → Serves static assets

## Error Handling System

DocExplore implements comprehensive error handling at multiple levels to ensure a smooth user experience even when things go wrong:

### 1. Frontend Validation

The first line of defense is client-side validation in the browser:

```javascript
// In upload.html - Frontend validation
form.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        showError('Please select a file to upload');
        return;
    }
    
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.pdf') && !file.name.toLowerCase().endsWith('.json')) {
        showError('File must be a PDF or JSON file');
        return;
    }
    
    // Continue with upload if validation passes
    // ...
});
```

This prevents invalid submissions before they reach the server.

### 2. Backend Validation

The server performs additional validation to catch any issues that bypass client-side checks:

```python
# In filehandler.py - Server-side validation
if not uploaded_files:
    # Handle case where no file was uploaded
    self.set_status(400)
    self.write({"error": "No file uploaded"})
    return

# Extract file information
file_info = uploaded_files[0]
file_name = file_info['filename']

# Validate that the file is a PDF
if not file_name.lower().endswith('.pdf'):
    self.set_status(400)
    self.write({"error": "File must be a PDF"})
    return
```

### 3. Processing Error Handling

During PDF processing, the application catches and handles errors that might occur:

```python
# In extract_pdf_to_json.py - Processing error handling
try:
    # Try to extract text from a page
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"
except Exception as e:
    # Log the error but continue with other pages
    print(f"Error extracting text from page {i}: {e}")
```

The `process_pdf_to_json` function contains multiple try-except blocks to handle errors at different stages:

```python
# In extract_pdf_to_json.py - Main function error handling
# Load SentenceTransformer model and compute embeddings
try:
    print(f"Loading SentenceTransformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = compute_embeddings(paragraphs, model)
except Exception as e:
    print(f"Error loading model or computing embeddings: {e}")
    return False  # Return failure status

# Cluster the embeddings into topics
try:
    labels, centroids = perform_clustering(embeddings, NUM_TOPICS)
    actual_topics = len(np.unique(labels))
except Exception as e:
    print(f"Error during clustering: {e}")
    return False  # Return failure status
```

### 4. Subprocess Error Handling

The backend handles errors that might occur when running the processing script:

```python
# In filehandler.py - Subprocess error handling
try:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ, PYTHONPATH=os.getcwd())
    )
    
    # Check if the output file exists
    if not os.path.exists(output_json_path):
        raise Exception("Output file was not created")
    
    # Verify the output is valid JSON
    with open(output_json_path, 'r', encoding='utf-8') as f:
        json.load(f)  # Will raise JSONDecodeError if invalid
    
except subprocess.CalledProcessError as e:
    logger.error(f"Script execution failed with return code {e.returncode}")
    logger.error(f"Script stdout: {e.stdout}")
    logger.error(f"Script stderr: {e.stderr}")
    self.set_status(500)
    self.write({
        "error": "Failed to process PDF",
        "details": f"Script error: {e.stderr or e.stdout or str(e)}"
    })
    return
    
except Exception as e:
    logger.error(f"Error processing PDF: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    self.set_status(500)
    self.write({
        "error": "Failed to process PDF",
        "details": f"Processing error: {str(e)}"
    })
    return
```

### 5. User Feedback

The application provides clear error feedback to users:

```javascript
// In upload.html - Error display function
function showError(message, title = 'Error Processing File', debugInfo = '') {
    // Hide upload form and processing display
    uploadArea.style.display = 'none';
    processingDiv.style.display = 'none';
    
    // Show and populate error message
    errorDiv.style.display = 'block';
    errorText.textContent = message;
    
    // Add debug information if available
    if (debugInfo) {
        errorDetails.style.display = 'block';
        debugInfoEl.style.display = 'block';
        debugInfoEl.textContent = debugInfo;
    } else {
        errorDetails.style.display = 'none';
        debugInfoEl.style.display = 'none';
    }
}
```

### 6. Error Flow

When an error occurs, the application:

1. Catches the error at the appropriate level
2. Logs details for debugging
3. Cleans up any temporary files
4. Returns an appropriate HTTP status code and error message
5. Displays a user-friendly error message in the interface
6. Provides options for the user to retry or return to the main page

This multi-layered approach ensures that errors are handled gracefully at every stage of processing.
