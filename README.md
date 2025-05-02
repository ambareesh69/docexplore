 # Document Explorer: PDF Analysis and Visualization Tool

   ## What is Document Explorer?

   Document Explorer is a specialized web application that processes PDF documents to extract, analyze, and visualize their content. Consider it as a document processing system that receives your PDF, extracts all text, identifies main topics through advanced analysis, and presents the results in an organized, interactive format.

   The system performs these key operations:
   - Accepts PDF file uploads through a user-friendly interface
   - Extracts and processes the textual content
   - Analyzes the text to identify main topics and related concepts
   - Groups similar content together using natural language processing techniques
   - Presents the organized information in a visual, interactive format

   ## Setup Instructions

   To set up Document Explorer on your system, follow these steps:

   1. **System Requirements**
      - Python 3.11 or newer
      - Node.js and npm
      - Gramex framework (a web application server)

   2. **Installation Process**
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

   3. **Starting the Application**
      ```bash
      gramex
      ```
      Once started, access Document Explorer by navigating to http://localhost:9988 in your web browser.

   ## Project Structure

   The Document Explorer project consists of the following files and directories:

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

   ## Predefined Variables

   The application uses several predefined variables that control its behavior:

   | Variable | Value | Location | Purpose |
   |----------|-------|----------|---------|
   | MODEL_NAME | "all-MiniLM-L6-v2" | extract_pdf_to_json.py | Specifies which SentenceTransformer model to use for text analysis - this model provides a good balance of speed and accuracy |
   | NUM_TOPICS | 3 | extract_pdf_to_json.py | Defines the default number of topic groups to identify in a document - this number represents a balance between detail and simplicity |
   | SIMILARITY_THRESHOLD | 0.75 | extract_pdf_to_json.py | Minimum similarity score (0-1) required to assign a paragraph to a topic - higher values create more distinct topic groups |
   | random_state | 42 | extract_pdf_to_json.py | Ensures consistent clustering results between runs - this fixed seed makes the analysis reproducible |

   ## Backend Processing Workflow

   ### 1. PDF File Upload

   **What Happens**: You upload a PDF file through the web interface.

   **How It Works**: The system receives your PDF file, validates it, and stores it for processing.

   **Files Involved**:
   - `upload.html`: Provides the interface for file selection
   - `story.js`: Handles file upload and progress tracking
   - `filehandler.py`: Processes the uploaded file
   - `gramex.yaml`: Routes the upload request to the appropriate handler

   **Processing Flow**:

   1. **Frontend Initiation**:
      The upload process begins in `story.js` when you submit a form or drop a file:

      ```javascript
      // When form is submitted or file is dropped
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
         
         // Send the file to the server
         fetch(endpoint, {
            method: 'POST',
            body: formData
         })
         // ...response handling follows
      });
      ```

   2. **Backend Reception**:
      The `PDFProcessHandler` in `filehandler.py` receives and processes the file:

      ```python
      # The PDF handling function that receives uploads
      async def post(self):
         # Get the session ID from the request or generate a new one
         session_id = self.get_argument('session_id', str(uuid.uuid4()))
         logger.info(f"Starting PDF processing for session: {session_id}")
         
         # Send initial progress update
         self.update_progress(session_id, 'upload', 0)
         
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
         
         # Continue to PDF processing stage
         # ...processing code follows
      ```

   **Flow Summary**:
   1. User selects or drops a PDF file in the browser
   2. JavaScript creates FormData with the file and a unique session ID
   3. File is sent to the `/process-pdf` endpoint
   4. Backend validates the file is a PDF
   5. Backend saves the file to the `uploads/` directory
   6. Progress updates are sent at 0%, 50%, and 100% of the upload stage

   ### 2. PDF Text Extraction and Analysis

   **What Happens**: After saving your PDF, the system extracts and analyzes its textual content.

   **How It Works**: The system launches a specialized script that opens the PDF, extracts text from each page, and processes it to identify main topics.

   **Files Involved**:
   - `filehandler.py`: Initiates the processing workflow
   - `backend/extract_pdf_to_json.py`: Performs the actual text extraction and analysis

   **Processing Flow**:

   1. **Initiating Text Extraction**:
      After the upload completes, `filehandler.py` continues with text extraction:

      ```python
      # Begin text extraction phase
      self.update_progress(session_id, 'extract', 0)
      
      # Locate the Python interpreter and processing script
      venv_python = os.path.join(os.getcwd(), 'gramex311', 'bin', 'python')
      if not os.path.exists(venv_python):
         raise Exception(f"Virtual environment Python not found at {venv_python}")
      
      script_path = os.path.abspath('backend/extract_pdf_to_json.py')
      if not os.path.exists(script_path):
         raise Exception(f"Script not found at {script_path}")
      
      # Update progress to indicate preparation complete
      self.update_progress(session_id, 'extract', 50)
      
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
      
      # Execute the extraction script as a subprocess
      result = subprocess.run(
         command,
         capture_output=True,  # Capture stdout and stderr
         text=True,  # Return strings instead of bytes
         check=True,  # Raise exception on non-zero exit
         env=dict(os.environ, PYTHONPATH=os.getcwd())  # Set environment variables
      )
      ```

   2. **PDF Text Extraction**:
      The `extract_pdf_to_json.py` script handles text extraction:

      ```python
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

   3. **Text Processing and Analysis**:
      After extraction, the text is split and processed:

      ```python
      def split_into_paragraphs(text):
         """Split text into paragraphs based on double newlines."""
         # Split the text wherever there are two newlines (paragraph breaks)
         paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
         print(f"Split text into {len(paragraphs)} paragraphs")
         return paragraphs
      
      def compute_embeddings(paragraphs, model):
         """Convert paragraphs to numerical embeddings using SentenceTransformer."""
         print(f"Computing embeddings for {len(paragraphs)} paragraphs")
         # Transform text into numerical vectors
         embeddings = model.encode(paragraphs)
         print(f"Computed embeddings with shape: {embeddings.shape}")
         return embeddings
      ```

   4. **Topic Identification through Clustering**:
      The system groups similar paragraphs using clustering:

      ```python
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

   5. **Topic Naming**:
      The system identifies representative keywords for each topic:

      ```python
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

   6. **JSON Output Generation**:
      All results are compiled into a structured JSON format:

      ```python
      # Build the topics array
      topics = []
      for i in range(actual_topics):
         # Get paragraphs belonging to this topic
         cluster_paragraphs = [paragraphs[j] for j in range(len(paragraphs)) if labels[j] == i]
         # Extract representative keyword
         keyword = extract_topic_keyword(cluster_paragraphs)
         topics.append({
            "topic_id": i,
            "topic": keyword,
            "subtopic": f"Subtopic of {keyword}",
            "count": len(cluster_paragraphs)
         })
      
      # Build the docs and matches arrays
      docs = []
      matches = []
      for j, para in enumerate(paragraphs):
         # Process each paragraph
         doc_embedding = embeddings[j]
         assigned = assign_topics_to_doc(doc_embedding, centroids, SIMILARITY_THRESHOLD)
         cluster_label = labels[j]
         
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
         
         # Add to matches array
         for topic_id, sim in assigned:
            matches.append({
                  "doc": j,
                  "topic": topic_id,
                  "similarity": sim
            })
      
      # Write JSON to file
      with open(output_json, "w", encoding="utf-8") as f:
         json.dump(output_data, f, indent=4)
      ```

   7. **Processing Completion**:
      After the script completes, `filehandler.py` verifies the results:

      ```python
      # Complete analyze phase
      self.update_progress(session_id, 'analyze', 100)
      
      # Begin process phase
      self.update_progress(session_id, 'process', 0)
      
      # Verify output file exists
      if not os.path.exists(output_json_path):
         raise Exception("Output file was not created")
      
      # Verify valid JSON
      with open(output_json_path, 'r', encoding='utf-8') as f:
         json.load(f)  # Will raise JSONDecodeError if invalid
      
      # Complete process phase
      self.update_progress(session_id, 'process', 100)
      
      # Mark processing as complete
      self.update_progress(session_id, 'complete', 100)
      
      # Send success response
      self.write({
         "status": "success",
         "message": "PDF processed successfully",
         "outputFile": os.path.basename(output_json_path)
      })
      ```

   **Flow Summary**:
   1. Backend initiates extraction and prepares processing command
   2. The extract_pdf_to_json.py script is executed as a subprocess
   3. Text is extracted from the PDF using PyPDF2
   4. Text is split into paragraphs
   5. SentenceTransformer converts paragraphs to numerical embeddings
   6. K-means clustering groups similar paragraphs
   7. TF-IDF identifies representative keywords for each topic
   8. Results are compiled into structured JSON
   9. Backend verifies output and sends completion response

   ### 3. Progress Tracking System

   **What Happens**: Throughout processing, you receive real-time updates about the status.

   **How It Works**: The system uses Server-Sent Events (SSE) to stream progress updates from the server to your browser.

   **Files Involved**:
   - `filehandler.py`: Manages the progress tracking and updates
   - `story.js`: Establishes the connection and handles updates
   - `upload.html`: Displays the progress to the user

   **Processing Flow**:

   1. **Frontend Connection Setup**:
      When you upload a file, `story.js` establishes an SSE connection:

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
                     processingMessage.textContent = `Uploading file... (${data.progress}%)`;
                     break;
                  case 'extract':
                     processingMessage.textContent = `Extracting text... (${data.progress}%)`;
                     break;
                  case 'analyze':
                     processingMessage.textContent = `Analyzing content... (${data.progress}%)`;
                     break;
                  case 'process':
                     processingMessage.textContent = `Processing insights... (${data.progress}%)`;
                     break;
                  case 'complete':
                     processingMessage.textContent = 'Process complete! Redirecting...';
                     break;
            }
         } catch (error) {
            console.error('Error parsing progress data:', error);
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
                        
                        # If process is complete, send closing event
                        if current_update.get('step') == 'complete':
                              await asyncio.sleep(0.5)
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
      The `PDFProcessHandler` updates progress at key points:

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

   **Flow Summary**:
   1. Frontend establishes SSE connection with session ID
   2. Backend sends initial "connection established" message
   3. As processing occurs, backend calls update_progress() at key points
   4. ProgressHandler continuously checks for new updates
   5. When updates occur, they're sent to the frontend
   6. Frontend updates the UI to reflect current status
   7. When processing completes, connection is closed and session data cleaned up

   ### 4. Result Presentation

   **What Happens**: After processing completes, you're redirected to view the analysis results.

   **How It Works**: The system loads the generated JSON and renders it in an interactive visualization.

   **Files Involved**:
   - `story.js`: Handles the redirection
   - `templates/insights.tmpl.html`: Renders the visualization
   - `gramex.yaml`: Routes the request to the insights template

   **Processing Flow**:

   1. **Frontend Redirection**:
      When processing completes, `story.js` redirects to the insights page:

      ```javascript
      .then(data => {
         console.log('Upload successful:', data);
         updateProgress('complete', 100);
         
         // Allow brief moment for final update
         setTimeout(() => {
            // Close the progress connection
            progressSource.close();
            // Redirect to insights page with the filename
            window.location.href = './insights?json=' + data.outputFile;
         }, 1000);
      })
      ```

   2. **Insights Template Rendering**:
      The `insights.tmpl.html` template loads and visualizes the JSON data.

   **Flow Summary**:
   1. Backend sends success response with output filename
   2. Frontend redirects to insights page with filename parameter
   3. Server renders insights template with the JSON data
   4. User sees interactive visualization of document topics and content

   ## Special Focus: PDF Analysis Technology

   The heart of Document Explorer is the PDF analysis technology in `extract_pdf_to_json.py`. This section explains in detail how the system analyzes your documents.

   ### SentenceTransformer: Converting Text to Numbers

   SentenceTransformer is an AI tool that converts text into numerical representations (embeddings) that preserve semantic meaning.

   ```python
   # Load the SentenceTransformer model
   model = SentenceTransformer(MODEL_NAME)

   # Convert paragraphs to embeddings
   embeddings = model.encode(paragraphs)
   ```

   **How It Works**:
   1. The model has been pre-trained on millions of text examples
   2. It learns to represent similar meanings with similar number patterns
   3. Each paragraph is converted to a vector (typically 384 numbers)
   4. These vectors capture the semantic meaning of the text

   For example, sentences like "The company increased revenue" and "Business earnings went up" would have similar embeddings despite using different words.

   ### Clustering: Finding Natural Groups in Text

   K-means clustering is a mathematical technique that groups similar items (in this case, paragraph embeddings) based on their proximity in the numerical space.

   ```python
   # Create K-means clustering model
   kmeans = KMeans(n_clusters=actual_clusters, random_state=42)

   # Assign each paragraph to a cluster
   labels = kmeans.fit_predict(embeddings)

   # Get the center point of each cluster
   centroids = kmeans.cluster_centers_
   ```

   **How It Works**:
   1. The system starts with a specified number of random center points
   2. Each paragraph is assigned to the nearest center
   3. Centers are moved to the average position of their assigned paragraphs
   4. Steps 2-3 repeat until the centers stabilize
   5. The final groupings represent topics in the document

   This process is similar to organizing books on shelves by subject, where similar books end up together.

   ### Topic Naming with TF-IDF

   To name each topic group, the system uses Term Frequency-Inverse Document Frequency (TF-IDF) to identify the most distinctive words.

   ```python
   # Create TF-IDF vectorizer
   vectorizer = TfidfVectorizer(stop_words='english')

   # Convert paragraphs to word importance matrix
   tfidf_matrix = vectorizer.fit_transform(paragraphs)

   # Get all words and their importance scores
   features = vectorizer.get_feature_names_out()
   scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()

   # Find the most important word
   top_index = int(np.argmax(scores))
   return features[top_index]
   ```

   **How It Works**:
   1. TF-IDF measures how unique a word is to a particular group
   2. Common words that appear everywhere (like "the" or "and") get low scores
   3. Words that appear frequently in one group but rarely in others get high scores
   4. The highest-scoring word becomes the topic name

   This ensures that topic names are representative of the unique content in each group.

   ## Routing Configuration

   Document Explorer uses the `gramex.yaml` file to define how requests are routed to different handlers:

   ```yaml
   url:
   # Main page route
   docexplore:
      pattern: /$YAMLURL/
      handler: FileHandler
      kwargs:
         path: upload.html
         template: true

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

   # Results page route
   insights:
      pattern: /$YAMLURL/insights
      handler: FileHandler
      kwargs:
         path: templates/insights.tmpl.html
         template: true
   ```

   This configuration acts as a traffic director, mapping URLs to specific handlers:
   - `/` → Shows the upload page
   - `/progress` → Streams progress updates
   - `/process-pdf` → Handles PDF uploads and processing
   - `/insights` → Displays analysis results

   ## Error Handling System

   Document Explorer implements comprehensive error handling at multiple levels:

   1. **Frontend Validation**:
      - Checks if a file is selected
      - Verifies file type is PDF
      - Provides immediate feedback for invalid selections

   2. **Backend Validation**:
      - Verifies file exists in the request
      - Validates file extension is .pdf
      - Checks for required environment components

   3. **Processing Error Handling**:
      - Catches and logs exceptions during PDF reading
      - Handles errors in text extraction for individual pages
      - Manages failures in model loading or processing

   4. **User Feedback**:
      - Detailed error messages explain what went wrong
      - Standard HTTP error codes indicate problem type
      - Errors are displayed prominently in the interface

   ## Notes

   1. **Environment Configuration**:
      - The application assumes Python is installed at `./gramex311/bin/python` on Unix/Mac systems or `.\gramex311\Scripts\python` on Windows. If your environment differs, you may need to modify paths in `filehandler.py`.

   2. **PDF Compatibility**:
      - Document Explorer works best with text-based PDFs.
      - Scanned documents without OCR may not extract properly.
      - Very large PDFs may require additional processing time.

   3. **Performance Considerations**:
      - The SentenceTransformer model requires significant memory to operate efficiently.
      - Default clustering is set to 3 topics but can be adjusted for more detailed analysis.
      - Initial model loading may cause a brief delay during first processing.

   ---

   Document Explorer is developed by [Gramener](https://gramener.com/), a data science company specializing in data visualization and insights.
