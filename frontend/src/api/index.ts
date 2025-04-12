
/**
 * 主进程与渲染进程通信频道定义
 * Definition of communication channels between main process and rendering process
 */
const ipcApiRoute = {
  test: 'controller/example/test',

  // os
  os: {
    pathExists: 'controller/os/pathExists',
    readdir: 'controller/os/readdir',
    readFile: 'controller/os/readFile',
    messageShow: 'controller/os/messageShow',
    messageShowConfirm: 'controller/os/messageShowConfirm',
    selectFolder: 'controller/os/selectFolder',
    selectPic: 'controller/os/selectPic',
    openDirectory: 'controller/os/openDirectory',
    loadViewContent: 'controller/os/loadViewContent',
    removeViewContent: 'controller/os/removeViewContent',
    createWindow: 'controller/os/createWindow',
    getWCid: 'controller/os/getWCid',
    sendNotification: 'controller/os/sendNotification',
    initPowerMonitor: 'controller/os/initPowerMonitor',
    getScreen: 'controller/os/getScreen',
    autoLaunch: 'controller/os/autoLaunch',
    setTheme: 'controller/os/setTheme',
    getTheme: 'controller/os/getTheme',
    window1ToWindow2: 'controller/os/window1ToWindow2',
    window2ToWindow1: 'controller/os/window2ToWindow1',
  },
  // melanoma analysis
  melanoma: {
    analyzeLesions: 'controller/melanoma/analyzeLesions',
  },
}

const specialIpcRoute = {
  appUpdater: 'custom/app/updater', // updater channel
}

export {
  ipcApiRoute,
  specialIpcRoute
}

