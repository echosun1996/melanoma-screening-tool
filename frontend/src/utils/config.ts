// config.ts
import packageJson from '../../../package.json';

export const APP_CONFIG = {
    // version: packageJson.version,
    version: process.env.APP_VERSION || 'v0.1.2', // 会被构建工具替换为真实值
    versionCheckUrl: 'https://gist.githubusercontent.com/echosun1996/381bb20275451ff9bceebed951a4d895/raw/gistfile1.txt'
};