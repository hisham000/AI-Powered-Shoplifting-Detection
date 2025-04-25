import { useState, useEffect } from 'react';

// Custom hook to manage processed files
const useProcessedFiles = () => {
  const [processedFiles, setProcessedFiles] = useState<Set<string>>(new Set());
  
  // Load processed files from localStorage on mount
  useEffect(() => {
    const storedFiles = localStorage.getItem('processed_files');
    if (storedFiles) {
      setProcessedFiles(new Set(JSON.parse(storedFiles)));
    }
  }, []);
  
  // Save processed files to localStorage whenever they change
  useEffect(() => {
    if (processedFiles.size > 0) {
      localStorage.setItem('processed_files', JSON.stringify([...processedFiles]));
    }
  }, [processedFiles]);
  
  // Check if a file has been processed
  const isProcessed = (filePath: string): boolean => {
    return processedFiles.has(filePath);
  };
  
  // Mark a file as processed
  const markAsProcessed = (filePath: string): void => {
    setProcessedFiles(prev => {
      const updated = new Set(prev);
      updated.add(filePath);
      return updated;
    });
  };
  
  // Clear processed files history
  const clearProcessedFiles = (): void => {
    setProcessedFiles(new Set());
    localStorage.removeItem('processed_files');
  };
  
  return {
    isProcessed,
    markAsProcessed,
    clearProcessedFiles,
    processedCount: processedFiles.size
  };
};

export default useProcessedFiles;