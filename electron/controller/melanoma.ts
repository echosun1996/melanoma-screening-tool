import { crossService } from '../service/cross';
import { logger } from 'ee-core/log';
/**
 * Melanoma Analysis Controller
 * @class
 */
class MelanomaController {
  async helloServer(): Promise<any> {
    try {
      logger.info(`Testing connection to Python server`);

      // Call the hello endpoint
      const data = await crossService.requestApi('pyapp', '/hello', {}, {
        method: 'get',
        timeout: 5000 // 5 second timeout for quick response
      });

      return data;
    } catch (error) {
      logger.error('Error connecting to Python server:', error);
      throw error;
    }
  }
  /**
   * Analyze lesion images for melanoma assessment
   */
  async analyzeLesions(args: { lesions: any[] }): Promise<any> {
    try {
      const { lesions } = args;

      if (!lesions || !Array.isArray(lesions) || lesions.length === 0) {
        return {
          status: 'error',
          message: 'Invalid or empty lesion data',
          lesions: []
        };
      }

      logger.info(`Received ${lesions.length} lesions for analysis, processing one by one`);

      // Process results array to hold all analysis results
      const processedLesions = [];

      // Process each lesion individually
      for (let i = 0; i < lesions.length; i++) {
        const lesion = lesions[i];

        logger.info(`Processing lesion ${i+1} of ${lesions.length}: ${lesion.id}`);

        // Create single lesion request data
        const requestData = {
          lesions: [lesion],
          requestMetadata: {
            appVersion: "1.0.0",
            device: "Electron App",
            timestamp: new Date().toISOString(),
            requestId: `req_${Date.now()}_${lesion.id}`,
            currentIndex: i,
            totalCount: lesions.length
          }
        };

        // Call Python service to analyze this lesion
        const result = await crossService.analyzeLesions(requestData);

        // If successful and contains result data
        if (result && result.lesions && result.lesions.length > 0) {
          // Add to processed results
          processedLesions.push(result.lesions[0]);
        }
      }

      // Return complete results
      return {
        status: 'success',
        message: 'All lesions analyzed successfully',
        lesions: processedLesions,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Error in melanoma analysis:', error);

      return {
        status: 'error',
        // @ts-ignore
        message: error.message || 'Analysis failed',
        lesions: []
      };
    }
  }
}

MelanomaController.toString = () => '[class MelanomaController]';

export default MelanomaController;