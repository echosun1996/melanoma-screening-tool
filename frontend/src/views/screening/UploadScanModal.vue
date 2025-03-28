const fetchLesionDat<template>
  <div v-if="isOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
    <div class="bg-white rounded-lg p-6 max-w-2xl w-full">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-bold">Select Patient Scan Files</h2>
        <button
            @click="$emit('close')"
            class="text-gray-500 hover:text-gray-700"
        >
          <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </button>
      </div>

      <!-- Step Indicators -->
      <div class="mb-4">
        <div class="flex justify-between">
          <div class="text-center">
            <div :class="[
              'rounded-full h-8 w-8 flex items-center justify-center mx-auto mb-1',
              currentStep >= 1 ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
            ]">1</div>
            <span class="text-sm">Select Path</span>
          </div>
          <div class="text-center">
            <div :class="[
              'rounded-full h-8 w-8 flex items-center justify-center mx-auto mb-1',
              currentStep >= 2 ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
            ]">2</div>
            <span class="text-sm">Select Patient</span>
          </div>
          <div class="text-center">
            <div :class="[
              'rounded-full h-8 w-8 flex items-center justify-center mx-auto mb-1',
              currentStep >= 3 ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
            ]">3</div>
            <span class="text-sm">Select Scan</span>
          </div>
          <div class="text-center">
            <div :class="[
              'rounded-full h-8 w-8 flex items-center justify-center mx-auto mb-1',
              currentStep >= 4 ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'
            ]">4</div>
            <span class="text-sm">Filter Lesions</span>
          </div>
        </div>
      </div>

      <!-- Step 1: Select Base Path -->
      <div v-if="currentStep === 1">
        <div class="one-block-1">
          <span>Select Data Storage Path</span>
        </div>
        <div class="one-block-2">
          <div class="flex items-center space-x-2">
            <div class="flex-grow">
              <input
                  v-model="basePath"
                  class="w-full p-2 border rounded"
                  readonly
                  placeholder="Please select a path..."
              />
            </div>
            <button
                @click="selectBasePath"
                class="px-4 py-2 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
            >
              Change Directory
            </button>
          </div>
        </div>

        <div class="mt-4">
          <div class="text-sm text-gray-500">
            <p>Default path: \\NAHWB360CAPT01\AHWB360CAPT01-DermaGraphix-IMG\</p>
            <p class="mt-2">Please select the root directory containing patient data</p>
          </div>
        </div>
      </div>

      <!-- Step 2: Select Patient Folder -->
      <div v-if="currentStep === 2">
        <div class="mb-2 flex items-center">
          <button
              @click="currentStep = 1"
              class="text-blue-500 hover:text-blue-700 mr-2"
          >
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <span class="font-medium">Base Path: {{ basePath }}</span>
        </div>

        <div class="mb-4">
          <div class="relative">
            <input
                v-model="searchQuery"
                type="text"
                placeholder="Search patient folders..."
                class="w-full p-2 border rounded pl-10"
                @input="searchFolders"
            />
            <svg class="h-5 w-5 absolute left-3 top-2.5 text-gray-400" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M21 21L15 15M17 10C17 13.866 13.866 17 10 17C6.13401 17 3 13.866 3 10C3 6.13401 6.13401 3 10 3C13.866 3 17 6.13401 17 10Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </div>

        <div class="mb-4 max-h-64 overflow-y-auto border rounded">
          <div v-if="isLoading" class="p-4 text-center text-gray-500">
            <svg class="animate-spin h-5 w-5 mx-auto mb-1" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Loading...
          </div>
          <div v-else-if="patientFolders.length === 0" class="p-4 text-center text-gray-500">
            No matching patient folders found
          </div>
          <div v-else>
            <div
                v-for="folder in patientFolders"
                :key="folder.name"
                class="p-3 hover:bg-gray-100 cursor-pointer border-b last:border-b-0 flex items-center"
                :class="{ 'bg-blue-50': selectedPatientFolder === folder.name }"
                @click="selectPatientFolder(folder.name)"
            >
              <svg class="h-5 w-5 mr-2 text-yellow-500" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 7V17C3 18.1046 3.89543 19 5 19H19C20.1046 19 21 18.1046 21 17V9C21 7.89543 20.1046 7 19 7H13L11 5H5C3.89543 5 3 5.89543 3 7Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
              </svg>
              <div class="font-medium">{{ folder.name }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 3: Select Scan Folder -->
      <div v-if="currentStep === 3">
        <div class="mb-2 flex items-center">
          <button
              @click="currentStep = 2"
              class="text-blue-500 hover:text-blue-700 mr-2"
          >
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <span class="font-medium">Patient Folder: {{ selectedPatientFolder }}</span>
        </div>

        <div class="mb-4 max-h-64 overflow-y-auto border rounded">
          <div v-if="isLoading" class="p-4 text-center text-gray-500">
            <svg class="animate-spin h-5 w-5 mx-auto mb-1" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            Loading...
          </div>
          <div v-else-if="scanFolders.length === 0" class="p-4 text-center text-gray-500">
            No scan folders found
          </div>
          <div v-else>
            <div
                v-for="folder in scanFolders"
                :key="folder.name"
                class="p-3 hover:bg-gray-100 cursor-pointer border-b last:border-b-0"
                :class="{ 'bg-blue-50': selectedScanFolder === folder.name }"
                @click="selectScanFolder(folder.name)"
            >
              <div class="flex items-center">
                <svg class="h-5 w-5 mr-2 text-blue-500" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M9 3H5C3.89543 3 3 3.89543 3 5V19C3 20.1046 3.89543 21 5 21H19C20.1046 21 21 20.1046 21 19V15M9 3L21 15M9 3V9H15L21 15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div class="font-medium">{{ formatTimestamp(folder.name) }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Step 4: Filter Lesions -->
      <div v-if="currentStep === 4">
        <div class="mb-2 flex items-center">
          <button
              @click="currentStep = 3"
              class="text-blue-500 hover:text-blue-700 mr-2"
          >
            <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M19 12H5M5 12L12 19M5 12L12 5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <span class="font-medium">Scan Folder: {{ formatTimestamp(selectedScanFolder) }}</span>
        </div>

        <div class="mb-4">
          <div class="text-sm font-medium mb-2">Filter Criteria:</div>
          <div class="grid grid-cols-2 gap-3">
            <div>
              <label class="block text-xs text-gray-700 mb-1">Major Axis (mm)</label>
              <div class="flex items-center">
                <span class="mr-2">≥</span>
                <input
                    v-model="filters.majorAxisMM"
                    type="number"
                    step="0.1"
                    min="0"
                    class="w-full p-1 border rounded text-sm"
                />
              </div>
            </div>
            <div>
              <label class="block text-xs text-gray-700 mb-1">deltaLBnorm</label>
              <div class="flex items-center">
                <span class="mr-2">≥</span>
                <input
                    v-model="filters.deltaLBnorm"
                    type="number"
                    step="0.1"
                    min="0"
                    class="w-full p-1 border rounded text-sm"
                />
              </div>
            </div>
            <div>
              <label class="block text-xs text-gray-700 mb-1">Boundary Score</label>
              <div class="flex items-center">
                <span class="mr-2">≤</span>
                <input
                    v-model="filters.outOfBoundsFraction"
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    class="w-full p-1 border rounded text-sm"
                />
              </div>
            </div>
            <div>
              <label class="block text-xs text-gray-700 mb-1">Lesion Confidence (%)</label>
              <div class="flex items-center">
                <span class="mr-2">≥</span>
                <input
                    v-model="filters.dnnLesionConfidence"
                    type="number"
                    step="1"
                    min="0"
                    max="100"
                    class="w-full p-1 border rounded text-sm"
                />
              </div>
            </div>
            <div>
              <label class="block text-xs text-gray-700 mb-1">Nevus Confidence (%)</label>
              <div class="flex items-center">
                <span class="mr-2">≥</span>
                <input
                    v-model="filters.neviConfidence"
                    type="number"
                    step="1"
                    min="0"
                    max="100"
                    class="w-full p-1 border rounded text-sm"
                />
              </div>
            </div>
          </div>
        </div>

        <div class="flex items-center justify-between mb-2">
          <div class="text-sm font-medium">
            Matching lesions: {{ filteredLesions.length }}
          </div>
          <div>
            <button
                @click="toggleSelectAll"
                class="text-sm text-blue-600 hover:text-blue-800"
            >
              {{ isAllSelected ? 'Deselect All' : 'Select All' }}
            </button>
          </div>
        </div>

        <div v-if="isLoading" class="p-4 text-center text-gray-500">
          <svg class="animate-spin h-5 w-5 mx-auto mb-1" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Parsing JSON data...
        </div>
        <div v-else-if="filteredLesions.length === 0" class="p-4 text-center text-gray-500">
          No lesions match the current criteria
        </div>
        <div v-else class="mb-4 max-h-64 overflow-y-auto border rounded">
          <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 p-2">
            <div
                v-for="(lesion, index) in filteredLesions"
                :key="index"
                class="border rounded-lg p-2 cursor-pointer hover:bg-gray-50"
                :class="{ 'ring-2 ring-blue-500': selectedLesions.includes(index) }"
                @click="toggleLesionSelection(index)"
            >
              <div class="relative">
                <img
                    :src="lesion.imageUrl"
                    alt="Lesion image"
                    class="w-full h-20 object-cover rounded"
                />
                <div class="absolute top-0 right-0 bg-white rounded-bl p-1">
                  <div v-if="selectedLesions.includes(index)" class="text-blue-500">
                    <svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M5 12L10 17L19 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                  </div>
                </div>
                <div class="absolute bottom-0 right-0 bg-black bg-opacity-70 text-white text-xs p-1 rounded-tl">
                  {{ (lesion.majorAxisMM || 0).toFixed(1) }}mm
                </div>
              </div>
              <div class="mt-1 text-xs">
                <div class="flex justify-between">
                  <span>Confidence:</span>
                  <span>{{ (lesion.dnnLesionConfidence || 0).toFixed(1) }}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="flex justify-between mt-4">
        <button
            @click="$emit('close')"
            class="px-4 py-2 text-gray-700 border rounded hover:bg-gray-50"
        >
          Cancel
        </button>

        <div>
          <button
              v-if="currentStep < 4"
              @click="nextStep"
              class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!canProceed"
          >
            Next
          </button>

          <button
              v-else
              @click="importSelectedLesions"
              class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="selectedLesions.length === 0"
          >
            <div class="flex items-center">
              <svg class="h-5 w-5 mr-1" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 16L8 12L10 14L16 8L20 12M4 20V18.5M20 20V18.5M18.5 4H20M18.5 20H5.5M5.5 4H7M4 5.5V7M20 5.5V7M3 12C3 16.9706 7.02944 21 12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
              Import Selected ({{ selectedLesions.length }})
            </div>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, computed, watch } from 'vue';
import { ipc } from '@/utils/ipcRenderer';
import { ipcApiRoute } from '@/api';
import { message } from 'ant-design-vue';

export default {
  name: 'UploadScanModal',
  props: {
    isOpen: {
      type: Boolean,
      required: true
    },
    lesionsCount: {
      type: Number,
      default: 0
    }
  },
  emits: ['close', 'submit'],
  setup(props, { emit }) {
    // Step state
    const currentStep = ref(1);
    const isLoading = ref(false);

    // Step 1: Base path selection
    const basePath = ref('\\\\NAHWB360CAPT01\\AHWB360CAPT01-DermaGraphix-IMG\\');

    // Step 2: Patient folder selection
    const searchQuery = ref('');
    const patientFolders = ref([]);
    const selectedPatientFolder = ref('');

    // Step 3: Scan selection
    const scanFolders = ref([]);
    const selectedScanFolder = ref('');

    // Step 4: Lesion filtering
    const lesionData = ref([]);
    const selectedLesions = ref([]);
    const filters = reactive({
      majorAxisMM: 2.0,
      deltaLBnorm: 4.5,
      outOfBoundsFraction: 0.25,
      dnnLesionConfidence: 50.0,
      neviConfidence: 80.0,
    });

    // Computed properties
    const canProceed = computed(() => {
      if (currentStep.value === 1) {
        return basePath.value !== '';
      } else if (currentStep.value === 2) {
        return selectedPatientFolder.value !== '';
      } else if (currentStep.value === 3) {
        return selectedScanFolder.value !== '';
      }
      return false;
    });

    const filteredLesions = computed(() => {
      return lesionData.value.filter(lesion =>
          lesion.majorAxisMM >= filters.majorAxisMM &&
          lesion.deltaLBnorm >= filters.deltaLBnorm &&
          lesion.outOfBoundsFraction <= filters.outOfBoundsFraction &&
          lesion.dnnLesionConfidence >= filters.dnnLesionConfidence &&
          lesion.neviConfidence >= filters.neviConfidence
      );
    });

    const isAllSelected = computed(() => {
      return filteredLesions.value.length > 0 &&
          selectedLesions.value.length === filteredLesions.value.length;
    });

    // Watch for filter criteria changes
    watch(filters, () => {
      // Reset selected lesions when filter criteria change
      selectedLesions.value = [];
    });

    // Methods
    const selectBasePath = () => {
      ipc.invoke(ipcApiRoute.os.selectFolder)
          .then((res) => {
            if (res) {
              basePath.value = res;
              message.info(`Selected path: ${res}`);
            }
          })
          .catch((error) => {
            console.error('Error selecting folder:', error);
            message.error('Failed to select directory');
          });
    };

    const searchFolders = () => {
      fetchPatientFolders();
    };

    const fetchPatientFolders = () => {
      isLoading.value = true;

      try {
        ipc.invoke(ipcApiRoute.os.readdir, basePath.value)
            .then((files) => {
              if (!files || files.error) {
                console.error('Error reading directory:', files?.error || 'No files returned');
                message.error('Failed to read patient directories');
                isLoading.value = false;
                return;
              }

              // Process files directly if no additional info is needed
              patientFolders.value = files
                  .filter(file => file.isDirectory)
                  .filter(folder =>
                      searchQuery.value === '' ||
                      folder.name.toLowerCase().includes(searchQuery.value.toLowerCase())
                  )
                  .map(folder => ({
                    name: folder.name,
                    modified: folder.modifiedTime || 'Unknown'
                  }));

              isLoading.value = false;
            })
            .catch(error => {
              console.error('Error reading directories:', error);
              message.error('Failed to read patient directories');
              isLoading.value = false;
            });
      } catch (error) {
        console.error('Exception in fetchPatientFolders:', error);
        message.error('An error occurred while loading patient folders');
        isLoading.value = false;
      }
    };

    const selectPatientFolder = (folderName) => {
      selectedPatientFolder.value = folderName;
    };

    const fetchScanFolders = () => {
      isLoading.value = true;

      try {
        const fullPath = `${basePath.value}/${selectedPatientFolder.value}`;

        ipc.invoke(ipcApiRoute.os.readdir, fullPath)
            .then((files) => {
              if (!files || files.error) {
                console.error('Error reading scan folders:', files?.error || 'No files returned');
                message.error('Failed to read scan folders');
                isLoading.value = false;
                return;
              }

              // Process scan folders
              scanFolders.value = files
                  .filter(file => file.isDirectory)
                  .map(folder => ({
                    name: folder.name,
                    modified: folder.modifiedTime || 'Unknown'
                  }));

              isLoading.value = false;
            })
            .catch(error => {
              console.error('Error reading scan folders:', error);
              message.error('Failed to read scan folders');
              isLoading.value = false;
            });
      } catch (error) {
        console.error('Exception in fetchScanFolders:', error);
        message.error('An error occurred while loading scan folders');
        isLoading.value = false;
      }
    };

    const selectScanFolder = (folderName) => {
      selectedScanFolder.value = folderName;
    };

    const fetchLesionData = async () => {
      isLoading.value = true;
      try {
        // Construct the JSON file path
        let jsonPath = `${basePath.value}/${selectedPatientFolder.value}/${selectedScanFolder.value}/analysis/lesion_data_1.2.4.json`;

        // Check if the file exists
        let fileExists = await ipc.invoke(ipcApiRoute.os.pathExists, jsonPath);

        if (!fileExists) {

          jsonPath = `${basePath.value}/${selectedPatientFolder.value}/${selectedScanFolder.value}/analysis/lesion_data_1.3.0.json`;
          fileExists = await ipc.invoke(ipcApiRoute.os.pathExists, jsonPath);
          if (!fileExists) {
            message.error('Required file lesion_data.json not found in the selected scan folder');
            isLoading.value = false;
            return false; // Return false to indicate failure
          }else{
            message.info("Load 1.3.0: "+jsonPath)
          }
        }
        else{
          message.info("Load 1.2.4: "+jsonPath)
        }

        // Read the JSON file
        const jsonContent = await ipc.invoke(ipcApiRoute.os.readFile, jsonPath, 'utf-8');

        // Parse the JSON data
        const jsonData = JSON.parse(jsonContent);

        message.info(jsonData)

        // Process lesion data
        if (jsonData && jsonData.root && jsonData.root.children) {
          lesionData.value = jsonData.root.children.map(child => ({
            dnnLesionConfidence: child.dnnLesionConfidence || 0,
            neviConfidence: child.neviConfidence || 0,
            majorAxisMM: child.majorAxisMM || 0,
            deltaLBnorm: child.deltaLBnorm || 0,
            outOfBoundsFraction: child.outOfBoundsFraction || 0,
            imageUrl: "data:image/png;base64,"+child.img64cc || ''
          }));

          return true; // Return true to indicate success
        } else {
          message.error('Invalid lesion data format in the JSON file');
          return false;
        }
      } catch (error) {
        console.error('Error fetching lesion data:', error);
        message.error('Failed to read or parse the lesion data file');
        return false;
      } finally {
        isLoading.value = false;
      }
    };

    const toggleLesionSelection = (index) => {
      const position = selectedLesions.value.indexOf(index);
      if (position !== -1) {
        // Remove if already selected
        selectedLesions.value.splice(position, 1);
      } else {
        // Add if not selected
        selectedLesions.value.push(index);
      }
    };
    const toggleSelectAll = () => {
      if (isAllSelected.value) {
        // Deselect all
        selectedLesions.value = [];
      } else {
        // Select all
        selectedLesions.value = Array.from({ length: filteredLesions.value.length }, (_, i) => i);
      }
    };

    const nextStep = async () => {
      try {
        if (currentStep.value === 1) {
          // Verify path exists before proceeding
          const exists = await ipc.invoke(ipcApiRoute.os.pathExists, basePath.value);
          if (exists) {
            // Path exists, proceed to select patient step
            fetchPatientFolders();
            currentStep.value = 2;
          } else {
            message.error('Selected path does not exist or is not accessible');
          }
        } else if (currentStep.value === 2) {
          // Proceed to select scan step
          fetchScanFolders();
          currentStep.value = 3;
        } else if (currentStep.value === 3) {
          // Proceed to filter lesions step only if lesion data is successfully loaded
          const success = await fetchLesionData();
          if (success) {
            currentStep.value = 4;
          }
          // If fetchLesionData returns false, we stay on the current step
        }
      } catch (error) {
        console.error('Error in nextStep:', error);
        message.error('An error occurred while proceeding to the next step');
      }
    };

    const importSelectedLesions = () => {
      if (selectedLesions.value.length > 0) {
        const lesionsToImport = selectedLesions.value.map((index, arrayIndex) => {
          const lesion = filteredLesions.value[index];

          // 默认病变置信度计算概率
          const probability = 100;

          // 创建新的病变对象
          return {
            id: arrayIndex + 1, // 重新从1开始编号
            image: lesion.imageUrl,
            location: `${selectedPatientFolder.value}`,
            location: `N/A`,
            probability: probability,
            skinClass: lesion.neviConfidence >= 80 ? "Melanocytic" : "Non-melanocytic",
            recommendedAction: probability >= 0.8 ? "Urgent Biopsy" :
                probability >= 0.6 ? "Urgent Review" :
                    probability >= 0.5 ? "Review" : "Monitor",
            dimensions: `${lesion.majorAxisMM.toFixed(1)}mm x ${(lesion.majorAxisMM * 0.8).toFixed(1)}mm`,
            asymmetry: probability * 0.9, // 模拟不对称性
            border: probability * 0.85,   // 模拟边界不规则性
            color: probability * 0.95,    // 模拟颜色变异
            bodyPosition: {
              region: "upper_back", // 默认区域
              x: (Math.random() - 0.5) * 0.1,
              y: (Math.random() - 0.5) * 0.2,
              z: -0.12
            },
            // 添加选择条件信息
            selectionCriteria: {
              majorAxisMM: lesion.majorAxisMM.toFixed(1),
              deltaLBnorm: lesion.deltaLBnorm.toFixed(1),
              outOfBoundsFraction: lesion.outOfBoundsFraction.toFixed(2),
              dnnLesionConfidence: lesion.dnnLesionConfidence.toFixed(0),
              neviConfidence: lesion.neviConfidence.toFixed(0)
            }
          };
        });

        // 发送所有选定的病变到父组件
        emit('submit', lesionsToImport);
        emit('close'); // 关闭模态框
      }
    };

    // Format timestamp to human-readable form
    const formatTimestamp = (timestamp) => {
      if (!timestamp) return '';

      // Parse from format "20240325104309" to year, month, day, hour, minute
      const year = timestamp.substring(0, 4);
      const month = timestamp.substring(4, 6);
      const day = timestamp.substring(6, 8);
      const hour = timestamp.substring(8, 10);
      const minute = timestamp.substring(10, 12);

      return `${year}-${month}-${day} ${hour}:${minute}`;
    };

    // Initialization

    return {
      currentStep,
      isLoading,
      basePath,
      searchQuery,
      patientFolders,
      selectedPatientFolder,
      scanFolders,
      selectedScanFolder,
      lesionData,
      selectedLesions,
      filters,
      canProceed,
      filteredLesions,
      isAllSelected,
      selectBasePath,
      searchFolders,
      selectPatientFolder,
      selectScanFolder,
      toggleLesionSelection,
      toggleSelectAll,
      nextStep,
      importSelectedLesions,
      formatTimestamp
    };
  }
};
</script>

<style>
/* Import CDN version of Tailwind CSS */
@import 'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css';

/* Custom styles */
.one-block-1 {
  font-size: 16px;
  padding-top: 10px;
}
.one-block-2 {
  padding-top: 10px;
}
</style>