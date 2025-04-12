import { crossService } from '../service/cross';
import { logger } from 'ee-core/log';

/**
 * Melanoma Analysis Controller
 * @class
 */
class MelanomaController {
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

      logger.info(`Received ${lesions.length} lesions for analysis`);

      // Call Python service to analyze lesions
      const result = await crossService.analyzeLesions(lesions);

      return result;
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