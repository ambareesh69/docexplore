import os
import json
import uuid
import logging
import tornado.web
from gramex.handlers import BaseHandler
import subprocess
import traceback
import asyncio
from tornado.web import stream_request_body

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global progress tracking
progress_updates = {}

class ProgressHandler(BaseHandler):
    """Handle Server-Sent Events for progress updates."""
    
    async def get(self):
        """Stream progress updates to the client."""
        # Set essential SSE headers
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('X-Accel-Buffering', 'no')  # Disable nginx buffering
        
        # Write initial event to establish connection
        self.write("event: ping\n")
        self.write("data: {\"message\": \"Connection established\"}\n\n")
        await self.flush()
        
        session_id = self.get_argument('session', None)
        if not session_id:
            self.write("event: error\n")
            self.write("data: {\"error\": \"No session ID provided\"}\n\n")
            await self.flush()
            return
        
        # Add an initial progress update if none exists
        if session_id not in progress_updates:
            progress_updates[session_id] = {
                'step': 'upload',
                'progress': 0
            }
        
        # Keep track of last update to avoid sending duplicates
        last_update = None
        
        try:
            # Send regular updates as long as the session exists
            for _ in range(1000):  # Limit to prevent infinite loops
                current_update = progress_updates.get(session_id)
                
                # Only send if there's a new update
                if current_update and current_update != last_update:
                    last_update = current_update.copy()
                    
                    # Send a proper SSE formatted event
                    self.write(f"event: progress\n")
                    self.write(f"data: {json.dumps(current_update)}\n\n")
                    await self.flush()
                    
                    logger.info(f"Sent progress update to client: {current_update}")
                    
                    # If process is complete, send one final event and exit
                    if current_update.get('step') == 'complete':
                        await asyncio.sleep(0.5)  # Give time for the client to receive the complete event
                        self.write("event: close\n")
                        self.write("data: {\"message\": \"Closing connection\"}\n\n")
                        await self.flush()
                        
                        # Clean up session
                        if session_id in progress_updates:
                            del progress_updates[session_id]
                        break
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in progress handler: {str(e)}")
            self.write("event: error\n")
            self.write(f"data: {{\"error\": \"Progress tracking failed: {str(e)}\"}}\n\n")
            await self.flush()

class PDFProcessHandler(BaseHandler):
    """Handle PDF file uploads and process them using the extract_pdf_to_json.py script."""
    
    def initialize(self, **kwargs):
        super(PDFProcessHandler, self).initialize(**kwargs)
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    
    def options(self):
        self.set_status(204)
        self.finish()
    
    def update_progress(self, session_id, step, progress=0):
        """Update progress for a given session."""
        if not session_id:
            return
            
        progress_updates[session_id] = {
            'step': step,
            'progress': progress
        }
        logger.debug(f"Progress updated for session {session_id}: {step} - {progress}%")
    
    async def post(self):
        input_pdf_path = None
        output_json_path = None
        session_id = None
        
        try:
            # Get the session ID from the form data
            session_id = self.get_argument('session_id', str(uuid.uuid4()))
            logger.info(f"Starting PDF processing for session: {session_id}")
            
            # Initial progress update
            self.update_progress(session_id, 'upload', 0)
            await asyncio.sleep(0.1)
            
            # Get the uploaded file
            uploaded_files = self.request.files.get('file', [])
            if not uploaded_files:
                self.set_status(400)
                self.write({"error": "No file uploaded"})
                return
            
            file_info = uploaded_files[0]
            file_name = file_info['filename']
            
            logger.info(f"Received PDF file: {file_name}")
            self.update_progress(session_id, 'upload', 50)
            await asyncio.sleep(0.1)
            
            if not file_name.lower().endswith('.pdf'):
                self.set_status(400)
                self.write({"error": "File must be a PDF"})
                return
            
            # Use the original filename
            base_name = os.path.splitext(file_name)[0]  # Get filename without extension
            # Clean the filename to ensure it's valid
            base_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '_', '-')]).strip()
            
            input_pdf_path = os.path.join('uploads', f"{base_name}.pdf")
            output_json_path = os.path.join('uploads', f"{base_name}.json")
            
            # If files already exist, delete them
            if os.path.exists(input_pdf_path):
                os.remove(input_pdf_path)
            if os.path.exists(output_json_path):
                os.remove(output_json_path)
            
            # Ensure uploads directory exists
            os.makedirs('uploads', exist_ok=True)
            
            # Save the uploaded PDF
            with open(input_pdf_path, 'wb') as f:
                f.write(file_info['body'])
            
            logger.info(f"Saved PDF to {input_pdf_path}")
            self.update_progress(session_id, 'upload', 100)
            await asyncio.sleep(0.1)
            
            self.update_progress(session_id, 'extract', 0)
            await asyncio.sleep(0.1)
            
            # Get the virtual environment Python path
            venv_python = os.path.join(os.getcwd(), 'docexplore311', 'bin', 'python')
            if not os.path.exists(venv_python):
                raise Exception(f"Virtual environment Python not found at {venv_python}")
            
            script_path = os.path.abspath('backend/extract_pdf_to_json.py')
            if not os.path.exists(script_path):
                raise Exception(f"Script not found at {script_path}")
            
            self.update_progress(session_id, 'extract', 50)
            await asyncio.sleep(0.1)
            
            input_path = os.path.abspath(input_pdf_path)
            output_path = os.path.abspath(output_json_path)
            
            # Create a command for the script using the virtual environment Python
            command = [
                venv_python,
                script_path,
                '--input', input_path,
                '--output', output_path
            ]
            
            logger.info(f"Running command: {' '.join(command)}")
            self.update_progress(session_id, 'analyze', 0)
            await asyncio.sleep(0.1)
            
            # Call the PDF to JSON script
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=dict(os.environ, PYTHONPATH=os.getcwd())
                )
                
                logger.info(f"Command stdout: {result.stdout}")
                logger.info(f"Command stderr: {result.stderr}")
                
                self.update_progress(session_id, 'analyze', 100)
                await asyncio.sleep(0.1)
                
                self.update_progress(session_id, 'process', 0)
                await asyncio.sleep(0.1)
                
                # Check if the output file was created
                if not os.path.exists(output_json_path):
                    raise Exception("Output file was not created")
                
                # Verify the output is valid JSON
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # This will raise JSONDecodeError if invalid
                
                self.update_progress(session_id, 'process', 100)
                await asyncio.sleep(0.1)
                
                # Set complete status before returning response
                self.update_progress(session_id, 'complete', 100)
                await asyncio.sleep(0.1)
                
                self.write({
                    "status": "success",
                    "message": "PDF processed successfully",
                    "outputFile": os.path.basename(output_json_path)
                })
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Script execution failed with return code {e.returncode}")
                logger.error(f"Script stdout: {e.stdout}")
                logger.error(f"Script stderr: {e.stderr}")
                self.set_status(500)
                self.write({
                    "error": "Failed to process PDF",
                    "details": f"Script error: {e.stderr or e.stdout or str(e)}"
                })
                # Clean up the input file
                if input_pdf_path and os.path.exists(input_pdf_path):
                    os.remove(input_pdf_path)
                return
                
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.set_status(500)
                self.write({
                    "error": "Failed to process PDF",
                    "details": f"Processing error: {str(e)}"
                })
                # Clean up the files
                if input_pdf_path and os.path.exists(input_pdf_path):
                    os.remove(input_pdf_path)
                if output_json_path and os.path.exists(output_json_path):
                    os.remove(output_json_path)
                return
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.exception("Error in PDFProcessHandler")
            self.set_status(500)
            self.write({
                "error": "An unexpected error occurred",
                "details": str(e),
                "trace": error_details
            })
            # Clean up any files that might have been created
            if input_pdf_path and os.path.exists(input_pdf_path):
                os.remove(input_pdf_path)
            if output_json_path and os.path.exists(output_json_path):
                os.remove(output_json_path)


class JSONProcessHandler(BaseHandler):
    """Handle JSON file uploads."""
    
    def initialize(self, **kwargs):
        super(JSONProcessHandler, self).initialize(**kwargs)
        # Allow cross-origin requests
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
    
    def options(self):
        # Respond to OPTIONS requests
        self.set_status(204)
        self.finish()
    
    async def post(self):
        try:
            # Get the uploaded file
            uploaded_files = self.request.files.get('file', [])
            if not uploaded_files:
                self.set_status(400)
                self.write({"error": "No file uploaded"})
                return
            
            file_info = uploaded_files[0]
            file_name = file_info['filename']
            
            logger.info(f"Received JSON file: {file_name}")
            
            if not file_name.lower().endswith('.json'):
                self.set_status(400)
                self.write({"error": "File must be a JSON"})
                return
            
            # Use the original filename
            base_name = os.path.splitext(file_name)[0]  # Get filename without extension
            # Clean the filename to ensure it's valid
            base_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '_', '-')]).strip()
            
            json_path = os.path.join('uploads', f"{base_name}.json")
            
            # If file already exists, delete it
            if os.path.exists(json_path):
                os.remove(json_path)
            
            # Ensure uploads directory exists
            os.makedirs('uploads', exist_ok=True)
            
            # Save the uploaded JSON
            with open(json_path, 'wb') as f:
                f.write(file_info['body'])
            
            logger.info(f"Saved JSON to {json_path}")
            
            # Verify it's valid JSON
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
                    # First check if the file is not empty
                    if not json_content.strip():
                        raise ValueError("Empty JSON file")
                    # Then parse it
                    json.loads(json_content)
                
                # Return the file path
                self.write({
                    "status": "success",
                    "message": "JSON processed successfully",
                    "outputFile": os.path.basename(json_path)
                })
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid JSON: {e}")
                os.remove(json_path)  # Clean up the invalid file
                self.set_status(400)
                self.write({
                    "error": "Invalid JSON file",
                    "details": str(e)
                })
            except Exception as e:
                logger.exception("Error validating JSON")
                self.set_status(500)
                self.write({
                    "error": "An unexpected error occurred",
                    "details": str(e)
                })
        except Exception as e:
            error_details = traceback.format_exc()
            logger.exception("Error in JSONProcessHandler")
            self.set_status(500)
            self.write({
                "error": "An unexpected error occurred",
                "details": str(e),
                "trace": error_details
            }) 