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