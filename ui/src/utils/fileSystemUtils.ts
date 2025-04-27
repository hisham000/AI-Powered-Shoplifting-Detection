import axios from 'axios';

// The backend server URL that will provide file system access
const FILE_SERVER_URL = process.env.REACT_APP_FILE_SERVER_URL || 'http://127.0.1:9000'; // File server URL
// The EEP service URL for health checks
const EEP_SERVICE_URL = process.env.REACT_APP_EEP_URL || 'http://127.0.0.1:8000';

// Maximum number of retries when EEP is offline
const MAX_RETRIES = 3;
// Time to wait between retries (in milliseconds)
const RETRY_DELAY = 2000;

/**
 * Checks if the EEP service is online and healthy
 *
 * @returns Boolean indicating if EEP is online and healthy
 */
export async function isEEPOnline(): Promise<boolean> {
  try {
    const response = await axios.get(`${EEP_SERVICE_URL}/health`, {
      timeout: 3000,
    });
    return (
      response.data?.status === 'healthy' ||
      response.data?.status === 'degraded'
    );
  } catch (error) {
    console.error('EEP health check failed:', error);
    return false;
  }
}

/**
 * Executes a function with retry logic if EEP is offline
 *
 * @param operation Function to execute
 * @param retries Number of retries remaining
 * @returns Result of the operation or throws an error after retries exhausted
 */
async function withEEPCheck<T>(
  operation: () => Promise<T>,
  retries = MAX_RETRIES
): Promise<T> {
  if (await isEEPOnline()) {
    return operation();
  }

  if (retries <= 0) {
    throw new Error('EEP service is offline and max retries reached');
  }

  console.log(
    `EEP is offline. Retrying in ${RETRY_DELAY}ms... (${retries} retries left)`
  );

  // Wait for the retry delay
  await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));

  // Try again with one fewer retry
  return withEEPCheck(operation, retries - 1);
}

/**
 * Reads video files from the CCTV directory using the server endpoint
 * This is needed because browsers cannot directly access the file system
 *
 * @returns Array of full file paths to video files
 */
export async function readVideoFilesFromDirectory(): Promise<string[]> {
  try {
    return await withEEPCheck(async () => {
      const response = await axios.get(`${FILE_SERVER_URL}/readDirectory`);
      return response.data.files || [];
    });
  } catch (error) {
    console.error('Error reading directory:', error);
    return [];
  }
}

/**
 * Retrieves a video file from a path via the server
 *
 * @param filePath Path to the video file
 * @returns File object or undefined if not available
 */
export async function getVideoFile(
  filePath: string
): Promise<File | undefined> {
  try {
    return await withEEPCheck(async () => {
      const response = await axios.get(`${FILE_SERVER_URL}/getFile`, {
        params: { path: filePath },
        responseType: 'blob',
      });

      // Create a File object from the blob
      const filename = filePath.split('/').pop() || 'video.mp4';
      return new File([response.data], filename, { type: 'video/mp4' });
    });
  } catch (error) {
    console.error('Error getting file:', error);
    return undefined;
  }
}

/**
 * Fetches changes in the CCTV directory since the last check
 *
 * @param lastCheckedTime Timestamp of the last check
 * @returns Array of new or modified files
 */
export async function getDirectoryChanges(
  lastCheckedTime: Date
): Promise<string[]> {
  try {
    return await withEEPCheck(async () => {
      const response = await axios.get(`${FILE_SERVER_URL}/getChanges`, {
        params: {
          since: lastCheckedTime.toISOString(),
        },
      });

      return response.data.changes || [];
    });
  } catch (error) {
    console.error('Error getting directory changes:', error);
    return [];
  }
}
