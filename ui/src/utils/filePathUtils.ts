/**
 * Utility functions for working with CCTV file paths
 * Expected format: CCTV/cam{i}/YYYY-MM-DD/HH/YYYY-MM-DD_HH-MM-SS_cam{i}.mp4
 */

// Types for the extracted information
export interface FilePathInfo {
  cameraId: string;
  date: string;
  hour: string;
  minute: string;
  second: string;
  timestamp: Date;
  isValid: boolean;
}

/**
 * Parse a CCTV file path and extract its components
 * @param filePath The file path to parse
 * @returns Object with extracted components or null if invalid
 */
export const parseFilePath = (filePath: string): FilePathInfo | null => {
  // Regular expression to match the expected format
  const regex = /\/cam(\d+)\/(\d{4}-\d{2}-\d{2})\/(\d{2})\/\2_(\d{2})-(\d{2})-(\d{2})_cam\1\.mp4$/;
  
  const match = filePath.match(regex);
  if (!match) {
    return null;
  }
  
  const [, cameraId, date, hour, minute, second] = match;
  
  // Create a timestamp from the extracted components
  const [year, month, day] = date.split('-').map(Number);
  const timestamp = new Date(year, month - 1, day, parseInt(hour), parseInt(minute), parseInt(second));
  
  return {
    cameraId,
    date,
    hour,
    minute,
    second,
    timestamp,
    isValid: true
  };
};

/**
 * Validates if a given file path matches the expected CCTV format
 * @param filePath The file path to validate
 * @returns True if valid, false otherwise
 */
export const isValidCCTVPath = (filePath: string): boolean => {
  return parseFilePath(filePath) !== null;
};

/**
 * Extracts the camera information from a file path
 * @param filePath The file path
 * @returns A string with camera information (e.g. "Camera 1")
 */
export const extractCameraInfo = (filePath: string): string => {
  const info = parseFilePath(filePath);
  if (info) {
    return `Camera ${info.cameraId}`;
  }
  
  // Fallback extraction in case the full path doesn't match the expected format
  const fallbackMatch = filePath.match(/cam(\d+)/);
  return fallbackMatch ? `Camera ${fallbackMatch[1]}` : 'Unknown Camera';
};

/**
 * Formats a timestamp from a file path into a human-readable date
 * @param filePath The file path
 * @returns Formatted date string
 */
export const formatTimestampFromPath = (filePath: string): string => {
  const info = parseFilePath(filePath);
  if (info) {
    return info.timestamp.toLocaleString();
  }
  return 'Unknown Date';
};