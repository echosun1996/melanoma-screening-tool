<!--<template>-->
<!--  <router-view v-if="appCanRun"/>-->
<!--  <div v-else class="version-error-container">-->
<!--    &lt;!&ndash; This will only show if showVersionMismatchModal() isn't used &ndash;&gt;-->
<!--    <div class="version-error">-->
<!--      <h2>Version Mismatch</h2>-->
<!--      <p>The current application version does not match the server version. Please update to the latest version before using.</p>-->
<!--    </div>-->
<!--  </div>-->
<!--</template>-->


<template>
  <router-view v-if="appCanRun"/>
  <div v-else class="version-error-container">
    <!-- This will only show if showVersionMismatchModal() isn't used -->
    <div class="version-error">
      <h2>Version Mismatch</h2>
      <p>The current application version ({{ APP_CONFIG.version }}) does not match the server version. Please update to the latest version before using.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { checkVersion, showVersionMismatchModal } from './utils/versionCheck';
import { APP_CONFIG } from './utils/config';
const appCanRun = ref(true);

onMounted(async () => {
  // Remove loading element first
  const loadingElement = document.getElementById('loadingPage');
  if (loadingElement) {
    loadingElement.remove();
  }

  // Check version and handle mismatch
  const isVersionValid = await checkVersion(APP_CONFIG.version);
  if (!isVersionValid) {
    appCanRun.value = false;
    showVersionMismatchModal();
  }
});
</script>
<style lang="less">
.version-error-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: rgba(0, 0, 0, 0.8);
  z-index: 9999;

  .version-error {
    background-color: white;
    padding: 20px;
    border-radius: 5px;
    max-width: 500px;
    text-align: center;

    h2 {
      color: #e74c3c;
    }
  }
}
</style>