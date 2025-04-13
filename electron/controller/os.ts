import fs from 'fs';
import path from 'path';
import { app as electronApp, dialog, shell } from 'electron';

import { windowService } from '../service/os/window';
import {logger} from "ee-core/log";
/**
 * example
 * @class
 */
class OsController {

  /**
   * All methods receive two parameters
   * @param args Parameters transmitted by the frontend
   * @param event - Event are only available during IPC communication. For details, please refer to the controller documentation
   */

  /**
   * Message prompt dialog box
   */
  messageShow(): string {
    dialog.showMessageBoxSync({
      type: 'info', // "none", "info", "error", "question" 或者 "warning"
      title: 'Custom Title',
      message: 'Customize message content',
      detail: 'Other additional information'
    })

    return 'Opened the message box';
  }

  /**
   * Read file contents
   * @param args File path or object containing path and encoding
   * @param encoding Optional encoding parameter if first argument is string
   * @returns File contents or error object
   */
  readFile(args: string | { path: string, encoding?: string }, encoding?: string): Promise<string | Buffer | { error: string }> {
    try {
      let filePath: string;
      let fileEncoding: string | null = null;

      // Handle different parameter formats
      if (typeof args === 'string') {
        filePath = args;
        fileEncoding = encoding || null;
      } else {
        filePath = args.path;
        fileEncoding = args.encoding || encoding || null;
      }

      if (!filePath) {
        return Promise.resolve({ error: 'No file path provided' });
      }

      // Check if file exists
      if (!fs.existsSync(filePath)) {
        return Promise.resolve({ error: `File does not exist: ${filePath}` });
      }

      // Read file with or without encoding
      if (fileEncoding) {
        const content = fs.readFileSync(filePath, 'utf-8');
        return Promise.resolve(content);
      } else {
        const content = fs.readFileSync(filePath);
        return Promise.resolve(content);
      }

    } catch (err) {
      logger.error('Error reading file:', err);
      return Promise.resolve({ error: `Failed to read file: ${err.message}` });
    }
  }
  /**
   * Write content to a file with proper error handling
   * @param args File path and content or object containing path, data and encoding
   * @param data Optional content parameter if first argument is string
   * @param encoding Optional encoding parameter if needed
   * @returns Success status object or error object
   */
  writeFile(args: any, data?: any, encoding?: string): Promise<{ success: boolean, error?: string }> {
    const logPrefix = '[OsController.writeFile]';

    try {
      // Filter out Event objects that might be passed by Electron IPC
      if (args && typeof args === 'object' && args.sender && args.sender.send) {
        logger.debug(`${logPrefix} Detected Event object as first argument, shifting parameters`);
        // This is an Electron event object
        if (arguments.length >= 3) {
          // If we have 3+ arguments, the structure might be (event, path, data, encoding)
          args = data;
          data = encoding;
          encoding = arguments[3];
        } else {
          // Skip the event object and check if the second argument is the actual data
          logger.warn(`${logPrefix} Received Event object but not enough parameters`);
          return Promise.resolve({ success: false, error: 'Invalid arguments: Event object received but not enough parameters' });
        }
      }

      let filePath: string;
      let fileData: string | Buffer;
      let fileEncoding: string | null = null;

      // Handle different parameter formats
      if (typeof args === 'string') {
        filePath = args;
        fileData = data;
        fileEncoding = encoding || null;
        logger.debug(`${logPrefix} String format - Path: ${filePath}`);
      } else if (args && typeof args === 'object') {
        filePath = args.path;
        fileData = args.data;
        fileEncoding = args.encoding || encoding || null;
        logger.debug(`${logPrefix} Object format - Path: ${filePath}`);
      } else {
        logger.warn(`${logPrefix} Invalid arguments type: ${typeof args}`);
        return Promise.resolve({ success: false, error: `Invalid arguments type: ${typeof args}` });
      }

      // Validate inputs
      if (!filePath) {
        logger.warn(`${logPrefix} No file path provided`);
        return Promise.resolve({ success: false, error: 'No file path provided' });
      }

      if (fileData === undefined || fileData === null) {
        logger.warn(`${logPrefix} No data provided for path: ${filePath}`);
        return Promise.resolve({ success: false, error: 'No data provided' });
      }

      // Check if fileData is valid (string or Buffer)
      if (typeof fileData !== 'string' && !Buffer.isBuffer(fileData)) {
        logger.error(`${logPrefix} Invalid data type: ${typeof fileData}, constructor: ${fileData.constructor?.name}`);
        return Promise.resolve({
          success: false,
          error: `Invalid data type: ${typeof fileData}. Must be string or Buffer.`
        });
      }

      // Create directory if it doesn't exist
      const dirPath = path.dirname(filePath);
      if (!fs.existsSync(dirPath)) {
        logger.debug(`${logPrefix} Creating directory: ${dirPath}`);
        fs.mkdirSync(dirPath, { recursive: true });
      }

      // Write file with or without encoding
      if (fileEncoding) {
        fs.writeFileSync(filePath, fileData, { encoding: fileEncoding as BufferEncoding });
      } else {
        fs.writeFileSync(filePath, fileData);
      }

      logger.debug(`${logPrefix} File written successfully: ${filePath}`);
      return Promise.resolve({ success: true });
    } catch (err) {
      logger.error(`${logPrefix} Error writing file:`, err);
      return Promise.resolve({
        success: false,
        error: `Failed to write file: ${err.message}`
      });
    }
  }

  /**
   * Check if a path exists
   * @param args Path to check or object containing path
   * @returns Boolean indicating if path exists
   */
  pathExists(args: string | { path: string }): boolean {
    try {
      // Handle both string and object input formats
      const pathToCheck = typeof args === 'string' ? args : args.path;

      if (!pathToCheck) {
        logger.warn('pathExists called with empty path');
        return false;
      }

      // Check if path exists
      return fs.existsSync(pathToCheck);
    } catch (err) {
      logger.error('Error checking if path exists:', err);
      return false;
    }
  }

  /**
   * Read directory contents
   * @param args Path of the directory to read
   * @returns Array of file/directory entries or error object
   */
  readdir(args: string | { path: string }): Promise<{ error?: string } | Array<{ name: string, isDirectory: boolean, path: string }>> {
    try {
      // Handle both string and object input formats
      const dirPath = typeof args === 'string' ? args : args.path;

      if (!dirPath) {
        return Promise.resolve({ error: 'No directory path provided' });
      }

      // Check if directory exists
      if (!fs.existsSync(dirPath)) {
        return Promise.resolve({ error: `Directory does not exist: ${dirPath}` });
      }

      // Check if path is a directory
      const stats = fs.statSync(dirPath);
      if (!stats.isDirectory()) {
        return Promise.resolve({ error: `Path is not a directory: ${dirPath}` });
      }

      // Read directory contents
      const files = fs.readdirSync(dirPath);

      // Get file info for each entry
      const fileInfos = files.map(file => {
        const filePath = path.join(dirPath, file);
        let isDir = false;

        try {
          isDir = fs.statSync(filePath).isDirectory();
        } catch (err) {
          logger.error(`Error getting stats for file ${filePath}:`, err);
        }

        return {
          name: file,
          isDirectory: isDir,
          path: filePath
        };
      });

      return Promise.resolve(fileInfos);
    } catch (err) {
      logger.error('Error reading directory:', err);
      // @ts-ignore
      return Promise.resolve({ error: `Failed to read directory: ${err.message}` });
    }
  }

  /**
   * Message prompt and confirmation dialog box
   */
  messageShowConfirm(): string {
    const res = dialog.showMessageBoxSync({
      type: 'info',
      title: 'Custom Title',
      message: 'Customize message content',
      detail: 'Other additional information',
      cancelId: 1, // Index of buttons used to cancel dialog boxes
      defaultId: 0, // Set default selected button
      buttons: ['confirm', 'cancel'], 
    })
    let data = (res === 0) ? 'click the confirm button' : 'click the cancel button';
  
    return data;
  }

  /**
   * Select Directory
   */
  selectFolder() {
    const filePaths = dialog.showOpenDialogSync({
      properties: ['openDirectory', 'createDirectory']
    });

    if (!filePaths) {
      return ""
    }

    return filePaths[0];
  } 

  /**
   * open directory
   */
  openDirectory(args: { id: any }): boolean {
    const { id } = args;
    if (!id) {
      return false;
    }
    let dir = '';
    if (path.isAbsolute(id)) {
      dir = id;
    } else {
      dir = electronApp.getPath(id);
    }

    shell.openPath(dir);
    return true;
  }

  /**
   * Select Picture
   */
  selectPic(): string | null {
    const filePaths = dialog.showOpenDialogSync({
      title: 'select pic',
      properties: ['openFile'],
      filters: [
        { name: 'Images', extensions: ['jpg', 'png', 'gif'] },
      ]
    });
    if (!filePaths) {
      return null
    }
    
    try {
      const data = fs.readFileSync(filePaths[0]);
      const pic =  'data:image/jpeg;base64,' + data.toString('base64');
      return pic;
    } catch (err) {
      console.error(err);
      return null;
    }
  }   

  /**
   * Open a new window
   */
  createWindow(args: any): any {
    const wcid = windowService.createWindow(args);
    return wcid;
  }
  
  /**
   * Get Window contents id
   */
  getWCid(args: any): any {
    const wcid = windowService.getWCid(args);
    return wcid;
  }

  /**
   * Realize communication between two windows through the transfer of the main process
   */
  window1ToWindow2(args: any): void {
    windowService.communicate(args);
    return;
  }

  /**
   * Realize communication between two windows through the transfer of the main process
   */
  window2ToWindow1(args: any): void {
    windowService.communicate(args);
    return;
  }

  /**
   * Create system notifications
   */
  sendNotification(args: { title?: string; subtitle?: string; body?: string; silent?: boolean }, event: any): boolean {
    const { title, subtitle, body, silent} = args;

    const options: any = {};
    if (title) {
      options.title = title;
    }
    if (subtitle) {
      options.subtitle = subtitle;
    }
    if (body) {
      options.body = body;
    }
    if (silent !== undefined) {
      options.silent = silent;
    }
    windowService.createNotification(options, event);

    return true
  }   
}
OsController.toString = () => '[class OsController]';
logger.info("Load controller:os.ts")
export default OsController;