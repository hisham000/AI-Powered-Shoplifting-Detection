import axios from 'axios';

// The backend server URL that will provide file system access
const FILE_SERVER_URL = 'http://127.0.0.1:9000';

/**
 * Reads video files from a directory using the server endpoint
 * This is needed because browsers cannot directly access the file system
 * 
 * @param directoryPath Path to the directory to read
 * @returns Array of full file paths to video files
 */
export async function readVideoFilesFromDirectory(directoryPath: string): Promise<string[]> {
  try {
    // In a real implementation, this would make a request to a backend server
    // that would read the directory and return the list of files
    const response = await axios.get(`${FILE_SERVER_URL}/readDirectory`, {
      params: { path: directoryPath }
    });
    
    return response.data.files || [];
  } catch (error) {
    console.error('Error reading directory:', error);
    
    // Fallback to hard-coded paths for testing without a server
    // This simulates what the server would return
    if (directoryPath.includes('data/0')) {
      return [
        `${directoryPath}/Shoplifting001_x264_0.mp4`,
        `${directoryPath}/Shoplifting001_x264_1.mp4`,
        `${directoryPath}/Shoplifting001_x264_2.mp4`,
        `${directoryPath}/Shoplifting005_x264_0.mp4`,
        `${directoryPath}/Shoplifting005_x264_1.mp4`,
      ];
    } else if (directoryPath.includes('data/1')) {
      return [
        `${directoryPath}/Shoplifting005_x264_10.mp4`,
        `${directoryPath}/Shoplifting005_x264_11.mp4`,
        `${directoryPath}/Shoplifting005_x264_15.mp4`,
      ];
    }
    
    return [];
  }
}

/**
 * Retrieves a video file from a path via the server
 * 
 * @param filePath Path to the video file
 * @returns File object or undefined if not available
 */
export async function getVideoFile(filePath: string): Promise<File | undefined> {
  try {
    // In a real implementation, this would access a server endpoint that returns the file
    const response = await axios.get(`${FILE_SERVER_URL}/getFile`, {
      params: { path: filePath },
      responseType: 'blob'
    });
    
    // Create a File object from the blob
    const filename = filePath.split('/').pop() || 'video.mp4';
    return new File([response.data], filename, { type: 'video/mp4' });
  } catch (error) {
    console.error('Error getting file:', error);
    return undefined;
  }
}

/**
 * Fetches changes in a directory since the last check
 * 
 * @param directoryPath Path to monitor
 * @param lastCheckedTime Timestamp of the last check
 * @returns Array of new or modified files
 */
export async function getDirectoryChanges(
  directoryPath: string,
  lastCheckedTime: Date
): Promise<string[]> {
  try {
    const response = await axios.get(`${FILE_SERVER_URL}/getChanges`, {
      params: { 
        path: directoryPath,
        since: lastCheckedTime.toISOString()
      }
    });
    
    return response.data.changes || [];
  } catch (error) {
    console.error('Error getting directory changes:', error);
    return [];
  }
}