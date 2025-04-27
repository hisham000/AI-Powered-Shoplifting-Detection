import axios from 'axios';
import { ProcessVideoResponse } from '../models/types';

// Use environment variable with REACT_APP_ prefix if available, otherwise use default
const EEP_SERVICE_URL = process.env.REACT_APP_EEP_URL || 'http://127.0.0.1:8000';
const FILE_SERVER_URL = process.env.REACT_APP_FILE_SERVER_URL || 'http://127.0.1:9000'; // File server URL
// const FILE_SERVER_URL = 'http://127.0.0.1:9000'; // File server URL

// Service for handling API calls to the EEP
export const apiService = {
  // Process a video file and check for shoplifting
  processVideo: async (videoPath: string, videoFile?: File): Promise<ProcessVideoResponse> => {
    try {
      // Create form data to match the expected format
      const formData = new FormData();
      
      // Extract filename from path
      const fileName = videoPath.split('/').pop() || 'video.mp4';
      
      // Generate a video ID based on the filename and timestamp
      const videoId = `${fileName.replace(/\.[^/.]+$/, '')}_${Date.now()}`;
      
      // Use the actual video file if provided, otherwise create a dummy placeholder
      if (videoFile) {
        // Using an actual video file
        formData.append('file', videoFile, fileName);
        console.log(`Sending actual video file for processing: ${videoPath}, ID: ${videoId}`);
      } else {
        // No real file available, create a dummy placeholder
        const dummyBlob = new Blob(['dummy video content'], { type: 'video/mp4' });
        formData.append('file', dummyBlob, fileName);
        formData.append('path_only', 'true'); // Indicate this is just a path
        console.log(`Sending video path for processing: ${videoPath}, ID: ${videoId} (simulation mode)`);
      }
      
      // Add the required video_id parameter
      formData.append('video_id', videoId);
      
      // Add sample_fps parameter (optional, default is 4)
      formData.append('sample_fps', '4');
      
      // Set a timeout for the request
      const response = await axios.post(`${EEP_SERVICE_URL}/process-video`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 20000, // 20 seconds timeout for video processing
      });
      
      // Map the API response to our internal format
      const result: ProcessVideoResponse = {
        shoplifting_detected: response.data.shoplifting_detected,
        video_path: videoPath,
        timestamp: new Date().toISOString(),
        camera_id: extractCameraId(videoPath),
        video_id: response.data.video_id
      };
      
      return result;
    } catch (error: any) {
      console.error('Error processing video:', error);
      
      // Check if it's a server error (500)
      if (error.response && error.response.status === 500) {
        console.log('Server error (500) when processing video. Using simulation mode.');
        // Return a simulated response
        return simulateProcessVideoResponse(videoPath);
      }
      
      // For other errors, also use simulation
      return simulateProcessVideoResponse(videoPath);
    }
  },
  
  // Send confirmation about shoplifting detection result
  confirmVideo: async (videoPath: string, isShoplifting: boolean): Promise<any> => {
    try {
      // Extract or generate a video ID from the path
      const pathParts = videoPath.split('/');
      const fileName = pathParts[pathParts.length - 1];
      const videoId = fileName.split('.')[0]; // Use filename without extension as ID
      
      const response = await axios.post(`${EEP_SERVICE_URL}/confirm-video`, {
        video_id: videoId,
        label: isShoplifting ? 1 : 0
      }, {
        timeout: 5000, // 5 seconds timeout
      });
      
      return response.data;
    } catch (error) {
      console.error('Error confirming video:', error);
      // Return simulated response for the UI to continue working
      return { status: 'confirmed', simulation: true };
    }
  },
  
  // Check if the EEP server is available
  checkServerStatus: async (): Promise<boolean> => {
    try {
      const response = await axios.get(`${EEP_SERVICE_URL}/health`, { 
        timeout: 3000 
      });
      return response.status === 200;
    } catch (error) {
      console.error('EEP server is not available:', error);
      return false;
    }
  },

  // Get a playable URL for a video file from the file server
  getVideoUrl: (videoPath: string): string => {
    // URL encode the path for safety
    const encodedPath = encodeURIComponent(videoPath);
    return `${FILE_SERVER_URL}/getFile?path=${encodedPath}`;
  }
};

// Helper function to extract camera ID from video path
function extractCameraId(path: string): string {
  // Extract camera ID from path
  // For our test data format, we'll extract from "Shoplifting001_x264_0.mp4"
  const cameraMatch = path.match(/Shoplifting\d+_x264_(\d+)\.mp4$/);
  if (cameraMatch && cameraMatch[1]) {
    return `cam${cameraMatch[1]}`;
  }
  
  // For directory structure CCTV/cam{i}/...
  const dirMatch = path.match(/cam(\d+)/);
  if (dirMatch && dirMatch[1]) {
    return `cam${dirMatch[1]}`;
  }
  
  return 'cam0'; // Default camera ID
}

// Simulate a process video response for when the EEP server fails
function simulateProcessVideoResponse(videoPath: string): ProcessVideoResponse {
  // For demo purposes, 50% chance of "detecting" shoplifting
  const shoplifting_detected = Math.random() > 0.5;
  
  // Extract filename from path
  const fileName = videoPath.split('/').pop() || 'video.mp4';
  
  // Generate a video ID
  const videoId = `${fileName.replace(/\.[^/.]+$/, '')}_${Date.now()}`;
  
  return {
    shoplifting_detected,
    video_path: videoPath,
    timestamp: new Date().toISOString(),
    camera_id: extractCameraId(videoPath),
    video_id: videoId,
    simulation: true // Flag to indicate this is a simulated response
  };
}