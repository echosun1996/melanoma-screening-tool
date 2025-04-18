// types.ts
interface VersionResponse {
    version: string;
}

// versionCheck.ts
import { APP_CONFIG } from './config';

export async function checkVersion(currentVersion: string): Promise<boolean> {
    try {
        const response = await fetch(APP_CONFIG.versionCheckUrl);
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.status}`);
        }

        const text = await response.text();
        // Assuming the remote file contains just the version string or JSON with a version property
        let remoteVersion: string;

        try {
            // Try to parse as JSON first
            const data = JSON.parse(text) as VersionResponse;
            remoteVersion = data.version;
        } catch {
            // If not JSON, assume it's just a plain version string
            remoteVersion = text.trim();
        }

        return currentVersion === remoteVersion;
    } catch (error) {
        console.error('Version check failed:', error);
        // You can decide whether to allow the app to run on error or not
        // Return true to allow, false to block
        return false; // Block by default on error
    }
}

// Function to show a modal when versions don't match
export function showVersionMismatchModal(): void {
    const modal = document.createElement('div');
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    modal.style.zIndex = '9999';
    modal.style.display = 'flex';
    modal.style.justifyContent = 'center';
    modal.style.alignItems = 'center';

    const modalContent = document.createElement('div');
    modalContent.style.backgroundColor = 'white';
    modalContent.style.padding = '20px';
    modalContent.style.borderRadius = '5px';
    modalContent.style.maxWidth = '500px';
    modalContent.style.textAlign = 'center';

    const heading = document.createElement('h2');
    heading.textContent = '版本不匹配';
    heading.style.color = '#e74c3c';

    const message = document.createElement('p');
    message.textContent = '当前应用版本与服务器版本不匹配，请更新至最新版本后再使用。';

    modalContent.appendChild(heading);
    modalContent.appendChild(message);
    modal.appendChild(modalContent);

    document.body.appendChild(modal);
}