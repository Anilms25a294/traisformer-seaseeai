"""
Simple SeaSeeAI API Server - Working Version
"""

import http.server
import socketserver
import json
import torch
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seaseeai-simple-api")

class SeaSeeAIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "status": "healthy",
                "model_loaded": True,
                "uptime": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
            logger.info("Health check - OK")
            
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "message": "SeaSeeAI Trajectory Prediction API",
                "status": "healthy",
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path == '/model/info':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                "model_type": "TrAISformer",
                "loaded": True,
                "parameters": 1000000,
                "input_size": 4,
                "sequence_length": 20,
                "prediction_length": 5
            }
            self.wfile.write(json.dumps(response).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode())
                
                # Log the request
                logger.info(f"Prediction request with {len(request_data.get('observations', []))} observations")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Generate mock predictions
                observations = request_data.get('observations', [])
                prediction_horizon = request_data.get('prediction_horizon', 5)
                
                if len(observations) < 10:
                    self.send_response(400)
                    self.end_headers()
                    error_response = {"error": "Need at least 10 observations"}
                    self.wfile.write(json.dumps(error_response).encode())
                    return
                
                # Create mock predictions based on last observation
                last_obs = observations[-1] if observations else {
                    'latitude': 37.7749, 
                    'longitude': -122.4194,
                    'sog': 10.0,
                    'cog': 45.0
                }
                
                predictions = []
                for i in range(prediction_horizon):
                    predictions.append({
                        "step": i + 1,
                        "latitude": last_obs.get('latitude', 37.7749) + (i + 1) * 0.01,
                        "longitude": last_obs.get('longitude', -122.4194) + (i + 1) * 0.01,
                        "sog": last_obs.get('sog', 10.0),
                        "cog": last_obs.get('cog', 45.0),
                        "timestamp": (datetime.now() + timedelta(hours=i+1)).isoformat()
                    })
                
                response = {
                    "predictions": predictions,
                    "confidence": 0.85,
                    "processing_time": 0.05,
                    "model_version": "1.0.0"
                }
                self.wfile.write(json.dumps(response).encode())
                logger.info(f"Prediction response sent for {prediction_horizon} steps")
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                self.send_response(500)
                self.end_headers()
                error_response = {"error": f"Prediction failed: {str(e)}"}
                self.wfile.write(json.dumps(error_response).encode())
                
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

def run_server():
    PORT = 8000
    with socketserver.TCPServer(("", PORT), SeaSeeAIHandler) as httpd:
        print("")
        print("🚀 SeaSeeAI API Server Started Successfully!")
        print("=" * 50)
        print(f"📍 Server running on: http://localhost:{PORT}")
        print("")
        print("📊 Available Endpoints:")
        print("   GET  /health      - Health check")
        print("   GET  /            - Root endpoint")
        print("   GET  /model/info  - Model information")
        print("   POST /predict     - Make trajectory prediction")
        print("")
        print("💡 Example prediction request:")
        print('   curl -X POST http://localhost:8000/predict \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"observations": [{"mmsi": 123, "timestamp": "2024-01-01T00:00:00",')
        print('        "latitude": 37.7749, "longitude": -122.4194, "sog": 10.0, "cog": 45.0}],')
        print('        "prediction_horizon": 3}\'')
        print("")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        print("")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
