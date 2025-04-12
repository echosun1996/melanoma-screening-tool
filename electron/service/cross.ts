import { logger } from 'ee-core/log';
import { getExtraResourcesDir, getLogDir } from 'ee-core/ps';
import path from 'path';
import axios from 'axios';
import { is } from 'ee-core/utils';
import { cross } from 'ee-core/cross';

/**
 * cross
 * @class
 */
class CrossService {

  info(): string {
    const pids = cross.getPids();
    logger.info('cross pids:', pids);

    let num = 1;
    pids.forEach(pid => {
      let entity = cross.getProc(pid);
      logger.info(`server-${num} name:${entity.name}`);
      logger.info(`server-${num} config:`, entity.config);
      num++;
    })

    return 'hello electron-egg';
  }

  getUrl(name: string): string {
    const serverUrl = cross.getUrl(name);
    return serverUrl;
  }

  killServer(type: string, name: string): void {
    if (type == 'all') {
      cross.killAll();
    } else {
      cross.killByName(name);
    }
  }

  /**
   * create python service
   * In the default configuration, services can be started with applications.
   * Developers can turn off the configuration and create it manually.
   */
  async createPythonServer(): Promise<void> {
    // method 1: Use the default Settings
    //const entity = await cross.run(serviceName);

    // method 2: Use custom configuration
    const serviceName = "python";
    const opt = {
      name: 'pyapp',
      cmd: path.join(getExtraResourcesDir(), 'py', 'pyapp'),
      directory: path.join(getExtraResourcesDir(), 'py'),
      args: ['--port=7074'],
      windowsExtname: true,
      appExit: true,
    }
    const entity = await cross.run(serviceName, opt);
    logger.info('server name:', entity.name);
    logger.info('server config:', entity.config);
    logger.info('server url:', entity.getUrl());
  }

  /**
   * Request API from cross service
   * Enhanced to support both GET and POST methods with configurable timeout
   */
  async requestApi(name: string, urlPath: string, params: any, options: any = {}): Promise<any> {
    const serverUrl = cross.getUrl(name);
    const apiUrl = serverUrl + urlPath;
    logger.info(`Requesting API: ${apiUrl}`);

    // Default options
    const defaultOptions = {
      method: 'get',
      timeout: 30000, // Increased timeout for larger requests
      proxy: false,
    };

    // Merge with custom options
    const requestOptions = {
      ...defaultOptions,
      ...options,
      url: apiUrl,
    };

    // Handle data based on method
    if (requestOptions.method.toLowerCase() === 'post') {
      requestOptions.data = params;
    } else {
      requestOptions.params = params;
    }

    logger.info('Request options:', requestOptions);

    try {
      const response = await axios(requestOptions);
      if (response.status === 200) {
        const { data } = response;
        return data;
      }
    } catch (error) {
      logger.error('API request error:', error);
      throw error;
    }

    return null;
  }

  /**
   * Analyze lesion images using Python backend
   */
  async analyzeLesions(lesionData: any[]): Promise<any> {
    try {
      logger.info(`Analyzing ${lesionData.length} lesions with Python backend`);

      // Call the Python backend using POST method for larger data
      const data = await this.requestApi('pyapp', '/analyze', lesionData, {
        method: 'post',
        timeout: 60000, // 60 seconds timeout for image processing
      });

      return data;
    } catch (error) {
      logger.error('Error analyzing lesions:', error);
      throw error;
    }
  }
}

CrossService.toString = () => '[class CrossService]';
const crossService = new CrossService();

export {
  CrossService,
  crossService
};