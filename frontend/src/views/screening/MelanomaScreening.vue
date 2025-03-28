<template>
  <div class="h-screen flex flex-col overflow-hidden">
    <!-- Header -->
    <div class="bg-white border-b p-4">
      <div class="flex items-center justify-between">
        <h1 class="text-2xl font-bold">AI-Enhanced Melanoma Screening</h1>
        <div class="flex space-x-4">
          <button
              class="px-4 py-2 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
              @click="isUploadModalOpen = true"
          >
            Open Scan
          </button>
          <button class="px-4 py-2 bg-blue-100 text-blue-800 rounded">
            Export Report
          </button>
          <button class="px-4 py-2 bg-blue-100 text-blue-800 rounded">
            Previous Scan
          </button>
        </div>
      </div>
    </div>

    <div class="flex-1 flex overflow-hidden">
      <!-- Left Panel -->
      <div class="w-64 border-r bg-gray-50 flex flex-col h-full">
        <div class="p-4 space-y-2">
          <div class="bg-white p-4 rounded shadow">
            <div class="text-2xl font-bold">{{ lesions.length }}</div>
            <div class="text-sm text-gray-600">Total Lesions</div>
          </div>
          <div class="bg-white p-4 rounded shadow">
            <div class="text-2xl font-bold text-amber-600">
              {{ highRiskLesions.length }}
            </div>
            <div class="text-sm text-gray-600">Flagged for Review</div>
          </div>
        </div>

        <div class="p-4 border-t">
          <h3 class="font-medium mb-2">Risk Threshold</h3>
          <input
              type="range"
              v-model="riskThresholdValue"
              class="w-full"
              min="0"
              max="100"
              step="1"
          />
          <div class="text-sm text-gray-600 mt-1">
            {{ Math.round(riskThreshold * 100) }}%
          </div>
        </div>

        <div class="p-4">
          <select
              class="w-full p-2 border rounded"
              v-model="filters.bodyRegion"
          >
            <option value="all">All Regions</option>
            <option value="Upper Back">Upper Back</option>
            <option value="Lower Back">Lower Back</option>
            <option value="Chest">Chest</option>
          </select>
        </div>

        <!-- 3D Body Model -->
        <div class="flex-1 p-4 flex flex-col">
          <div
              class="relative flex-grow bg-gray-100 rounded-lg overflow-hidden"
              style="height: 300px"
          >
            <canvas
                ref="canvasRef"
                class="w-full h-full"
            />
            <div
                v-if="selectedLesion"
                class="absolute bottom-2 left-2 bg-white p-2 rounded shadow-md text-xs"
            >
              <p class="font-bold">Selected: {{ selectedLesion.location }}</p>
              <p class="text-red-600 font-bold">
                Risk: {{ (selectedLesion.probability * 100).toFixed(1) }}%
              </p>
            </div>
            <div class="absolute top-2 right-2 bg-white p-1 rounded-full shadow-md">
              <button
                  class="text-gray-600 hover:text-gray-800"
                  @click="resetModelView"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Center Panel -->
      <div class="flex-1 flex flex-col overflow-hidden">
        <div
            v-if="highRiskLesions.length > 0"
            class="bg-red-50 border border-red-200 text-red-700 p-4 m-4 mb-0 rounded flex items-center"
        >
          <svg class="h-4 w-4 mr-2" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
            <path d="M12 8V12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <span>
            {{ highRiskLesions.length }} lesion{{ highRiskLesions.length !== 1 ? 's' : '' }} require review
          </span>
        </div>

        <div class="flex-1 overflow-auto p-4">
          <div class="grid grid-cols-2 gap-4">
            <div
                v-for="lesion in visibleLesions"
                :key="lesion.id"
                class="p-4 rounded-lg border cursor-pointer transition-colors bg-gray-50 hover:shadow-lg"
                :class="{ 'ring-2 ring-blue-500': selectedLesion?.id === lesion.id }"
                @click="handleLesionClick(lesion)"
            >
              <div class="flex space-x-4">
                <div class="relative">
                  <img
                      :src="lesion.image"
                      :alt="`Lesion ${lesion.id}`"
                      class="object-cover rounded cursor-pointer w-32 h-32"
                      @click.stop="handleLesionClick(lesion)"
                  />
                  <button
                      class="absolute top-1 right-1 p-1 bg-white rounded-full shadow hover:bg-gray-100"
                      @click.stop="enlargedImage = lesion.image"
                  >
                    <svg class="h-4 w-4" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M15 15L21 21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                      <circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="2"/>
                      <path d="M10 7V13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                      <path d="M7 10H13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                  </button>
                </div>
                <div class="flex-1">
                  <div class="flex justify-between items-start mb-2">
                  <span
                      class="px-2 py-1 rounded text-sm font-medium"
                      :class="lesion.probability >= riskThreshold
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'"
                  >
                     {{ (lesion.probability * 100).toFixed(1) }}% Risk
                    </span>
                  </div>
                  <p class="text-sm text-gray-600">Location: {{ lesion.location }}</p>
                  <p class="text-sm text-gray-600">Size: {{ lesion.dimensions }}</p>
                  <p class="text-sm text-gray-600">Class: {{ lesion.skinClass }}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Panel -->
      <div v-if="showRightPanel" class="w-96 border-l bg-gray-50 overflow-auto">
        <div v-if="selectedLesion" class="p-4">
          <h2 class="text-lg font-bold mb-4">Detailed Analysis</h2>
          <div class="space-y-4">
            <img
                :src="selectedLesion.image"
                :alt="`Lesion ${selectedLesion.id}`"
                class="w-full rounded"
            />
            <div class="bg-white p-4 rounded shadow">
              <h3 class="font-medium mb-2">ABCD Analysis</h3>
              <div class="grid grid-cols-2 gap-2 text-sm">
                <div>Asymmetry:</div>
                <div class="text-right">{{ selectedLesion.asymmetry !== undefined ? (selectedLesion.asymmetry * 100).toFixed(1) + '%' : 'N/A' }}</div>
                <div>Border:</div>
                <div class="text-right">{{ selectedLesion.border !== undefined ? (selectedLesion.border * 100).toFixed(1) + '%' : 'N/A' }}</div>
                <div>Color Variation:</div>
                <div class="text-right">{{ selectedLesion.color !== undefined ? (selectedLesion.color * 100).toFixed(1) + '%' : 'N/A' }}</div>
                <div>Dimensions:</div>
                <div class="text-right">{{ selectedLesion.dimensions || 'N/A' }}</div>
              </div>
            </div>

            <div class="bg-white p-4 rounded shadow">
              <h3 class="font-medium mb-2">Management Recommendations</h3>
              <div class="space-y-3">
                <div
                    class="p-3 rounded"
                    :class="selectedLesion.probability >= 0.8
                    ? 'bg-red-50 border border-red-200'
                    : selectedLesion.probability >= 0.6
                      ? 'bg-yellow-50 border border-yellow-200'
                      : 'bg-green-50 border border-green-200'"
                >
                  <p class="font-medium mb-1">
                    {{ selectedLesion.probability >= 0.8
                      ? 'Urgent Action Required'
                      : selectedLesion.probability >= 0.6
                          ? 'Further Examination Needed'
                          : 'Routine Monitoring' }}
                  </p>
                  <p class="text-sm">
                    {{ selectedLesion.probability >= 0.8
                      ? 'Immediate biopsy recommended due to high-risk features. Schedule urgent dermatology consultation.'
                      : selectedLesion.probability >= 0.6
                          ? 'Dermoscopic examination recommended. Schedule within 2 weeks for detailed assessment.'
                          : 'Document and monitor. Review at next routine skin check.' }}
                  </p>
                </div>

                <div class="text-sm space-y-2">
                  <h4 class="font-medium">Recommended Actions:</h4>
                  <ul class="list-disc pl-4 space-y-1">
                    <template v-if="selectedLesion.probability >= 0.8">
                      <li>Schedule immediate biopsy</li>
                      <li>Full-body photography for baseline</li>
                      <li>Consider additional imaging if needed</li>
                      <li>Urgent dermatology referral</li>
                    </template>
                    <template v-else-if="selectedLesion.probability >= 0.6">
                      <li>Arrange dermoscopic examination</li>
                      <li>Compare with previous images if available</li>
                      <li>Schedule follow-up in 2-4 weeks</li>
                      <li>Document any changes in appearance</li>
                    </template>
                    <template v-else>
                      <li>Photograph for baseline reference</li>
                      <li>Schedule routine follow-up in 3-6 months</li>
                      <li>Patient education on self-monitoring</li>
                      <li>Document current appearance</li>
                    </template>
                  </ul>
                </div>

                <div v-if="selectedLesion.probability >= 0.6" class="text-sm space-y-2">
                  <h4 class="font-medium">Key Features of Concern:</h4>
                  <ul class="list-disc pl-4 space-y-1">
                    <li v-if="selectedLesion.asymmetry > 0.7">
                      Significant asymmetry detected
                    </li>
                    <li v-if="selectedLesion.border > 0.7">
                      Irregular border characteristics
                    </li>
                    <li v-if="selectedLesion.color > 0.7">
                      Concerning color variations
                    </li>
                    <li v-if="parseFloat(selectedLesion.dimensions) > 4">
                      Large diameter/size
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div class="bg-white p-4 rounded shadow mt-4">
              <h3 class="font-medium mb-2">Selection Criteria</h3>
              <div class="grid grid-cols-2 gap-2 text-sm">
                <div>Major Axis:</div>
                <div class="text-right">{{ selectedLesion.selectionCriteria?.majorAxisMM || 'N/A' }}</div>
                <div>Delta LB Norm:</div>
                <div class="text-right">{{ selectedLesion.selectionCriteria?.deltaLBnorm || 'N/A' }}</div>
                <div>Boundary Score:</div>
                <div class="text-right">{{ selectedLesion.selectionCriteria?.outOfBoundsFraction || 'N/A' }}</div>
                <div>Lesion Confidence:</div>
                <div class="text-right">{{ selectedLesion.selectionCriteria?.dnnLesionConfidence ? selectedLesion.selectionCriteria.dnnLesionConfidence + '%' : 'N/A' }}</div>
                <div>Nevus Confidence:</div>
                <div class="text-right">{{ selectedLesion.selectionCriteria?.neviConfidence ? selectedLesion.selectionCriteria.neviConfidence + '%' : 'N/A' }}</div>
              </div>
            </div>

            <div class="space-y-2 mt-4">
              <h3 class="font-medium">Clinical Notes</h3>
              <textarea
                  class="w-full p-2 border rounded"
                  rows="3"
                  placeholder="Add clinical notes..."
              ></textarea>
            </div>

            <div class="flex space-x-2 mt-4">
              <button class="flex-1 px-4 py-2 bg-green-100 text-green-800 rounded hover:bg-green-200">
                Mark Reviewed
              </button>
              <button class="flex-1 px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200">
                Flag for Biopsy
              </button>
            </div>
          </div>
        </div>
        <div v-else class="p-4 text-gray-500">
          Select a lesion to view detailed analysis
        </div>
      </div>
    </div>

    <!-- Enlarged Image Modal -->
    <div
        v-if="enlargedImage"
        class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
        @click="enlargedImage = null"
    >
      <div class="relative bg-white p-2 rounded-lg max-w-3xl max-h-3xl">
        <button
            class="absolute top-2 right-2 p-1 bg-white rounded-full shadow hover:bg-gray-100"
            @click="enlargedImage = null"
        >
          <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
        <img
            :src="enlargedImage"
            alt="Enlarged lesion"
            class="max-h-[80vh] object-contain"
        />
      </div>
    </div>

    <!-- Upload Scan Modal -->
    <UploadScanModal
        :is-open="isUploadModalOpen"
        :lesions-count="lesions.length"
        @close="isUploadModalOpen = false"
        @submit="handleNewLesion"
    />
  </div>
</template>

<script>
import { ref, reactive, computed, onMounted, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import gsap from 'gsap';
import UploadScanModal from './UploadScanModal.vue';


export default {
  name: 'MelanomaScreening',
  components: {
    UploadScanModal
  },
  setup() {
    // State
    const lesions = ref([]);
    const riskThresholdValue = ref(50);
    const riskThreshold = computed(() => riskThresholdValue.value / 100);
    const selectedLesion = ref(null);
    const viewMode = ref('triage');
    const currentPage = ref(1);
    const filters = reactive({
      bodyRegion: 'all',
      minSize: 0,
      maxSize: 100,
    });
    const showRightPanel = ref(false);
    const enlargedImage = ref(null);
    const isUploadModalOpen = ref(false);

    // THREE.js refs
    const canvasRef = ref(null);
    const scene = ref(null);
    const camera = ref(null);
    const renderer = ref(null);
    const controls = ref(null);
    const model = ref(null);
    const highlightMeshes = ref({});

    const ITEMS_PER_PAGE = 50;

    // Computed properties
    const processedLesions = computed(() => {
      return [...lesions.value]
          .sort((a, b) => b.probability - a.probability)
          .filter(lesion => {
            if (filters.bodyRegion !== 'all' && lesion.location !== filters.bodyRegion) return false;
            const [width] = lesion.dimensions.split('x').map(d => parseFloat(d));
            return width >= filters.minSize && width <= filters.maxSize;
          });
    });

    const highRiskLesions = computed(() => {
      return processedLesions.value.filter(l => l.probability >= riskThreshold.value);
    });

    const visibleLesions = computed(() => {
      const startIndex = (currentPage.value - 1) * ITEMS_PER_PAGE;
      return viewMode.value === 'triage'
          ? highRiskLesions.value
          : processedLesions.value.slice(startIndex, startIndex + ITEMS_PER_PAGE);
    });

    // Methods
    const handleLesionClick = (lesion) => {
      if (selectedLesion.value?.id === lesion.id) {
        showRightPanel.value = !showRightPanel.value;
      } else {
        selectedLesion.value = lesion;
        showRightPanel.value = true;
      }
    };

    const handleNewLesion = (newLesion) => {
      // 如果参数是数组，说明有多个病变要导入
      if (Array.isArray(newLesion)) {
        // 清空之前的lesions列表，只保留新上传的数据
        lesions.value = [...newLesion];

        // 选择第一个病变显示在右侧面板
        if (newLesion.length > 0) {
          selectedLesion.value = newLesion[0];
          showRightPanel.value = true;
        }
      } else {
        // 处理单个病变的情况 (向后兼容)
        lesions.value = [newLesion]; // 用新病变替换所有原有病变
        selectedLesion.value = newLesion;
        showRightPanel.value = true;
      }

      // 关闭上传模态框
      isUploadModalOpen.value = false;
    };

    const resetModelView = () => {
      if (model.value) {
        // Reset view to front
        gsap.to(model.value.rotation, {
          y: 0,
          duration: 0.5
        });

        if (camera.value && controls.value) {
          // Reset camera position
          gsap.to(camera.value.position, {
            x: 0,
            y: 0.8,
            z: 3,
            duration: 0.5,
            onComplete: () => {
              camera.value.lookAt(0, 0, 0);
              controls.value.update();
            }
          });
        }
      }
    };

    onMounted(() => {
      // 初始化一个空的病变列表，而不是使用样本数据
      lesions.value = [];

      // 初始化THREE.js
      initThreeJs();
    });

    // Lifecycle hooks
    // onMounted(() => {
    //   // Initialize sample data
    //   lesions.value = [
    //     {
    //       id: 1,
    //       image: "src/assets/images/image1.png",
    //       location: "Upper Back",
    //       probability: 0.89,
    //       skinClass: "Melanocytic",
    //       recommendedAction: "Urgent Review",
    //       dimensions: "4.8mm x 3.2mm",
    //       asymmetry: 0.76,
    //       border: 0.82,
    //       color: 0.91,
    //       bodyPosition: {
    //         region: "upper_back",
    //         x: 0,
    //         y: 0.25,
    //         z: -0.12
    //       }
    //     },
        // {
        //   id: 2,
        //   image: "src/assets/images/image2.png",
        //   location: "Upper Back",
        //   probability: 0.72,
        //   skinClass: "Melanocytic",
        //   recommendedAction: "Review",
        //   dimensions: "3.9mm x 3.1mm",
        //   asymmetry: 0.65,
        //   border: 0.71,
        //   color: 0.68,
        //   bodyPosition: {
        //     region: "upper_back",
        //     x: -0.05,
        //     y: 0.2,
        //     z: -0.12
        //   }
        // },
        // {
        //   id: 3,
        //   image: "src/assets/images/image3.png",
        //   location: "Lower Back",
        //   probability: 0.91,
        //   skinClass: "Melanocytic",
        //   recommendedAction: "Urgent Biopsy",
        //   dimensions: "5.2mm x 4.1mm",
        //   asymmetry: 0.88,
        //   border: 0.85,
        //   color: 0.93,
        //   bodyPosition: {
        //     region: "lower_back",
        //     x: 0.02,
        //     y: 0,
        //     z: -0.12
        //   }
        // },
        // {
        //   id: 4,
        //   image: "src/assets/images/image4.png",
        //   location: "Chest",
        //   probability: 0.45,
        //   skinClass: "Non-melanocytic",
        //   recommendedAction: "Monitor",
        //   dimensions: "3.1mm x 2.8mm",
        //   asymmetry: 0.42,
        //   border: 0.38,
        //   color: 0.51,
        //   bodyPosition: {
        //     region: "chest",
        //     x: -0.03,
        //     y: 0.4,
        //     z: 0.12
        //   }
        // },
        // {
        //   id: 5,
        //   image: "src/assets/images/image5.png",
        //   location: "Upper Back",
        //   probability: 0.83,
        //   skinClass: "Melanocytic",
        //   recommendedAction: "Urgent Review",
        //   dimensions: "4.5mm x 3.8mm",
        //   asymmetry: 0.79,
        //   border: 0.77,
        //   color: 0.85,
        //   bodyPosition: {
        //     region: "upper_back",
        //     x: 0.07,
        //     y: 0.3,
        //     z: -0.12
        //   }
        // },
        // {
        //   id: 6,
        //   image: "src/assets/images/image6.png",
        //   location: "Lower Back",
        //   probability: 0.67,
        //   skinClass: "Melanocytic",
        //   recommendedAction: "Review",
        //   dimensions: "3.7mm x 3.2mm",
        //   asymmetry: 0.61,
        //   border: 0.65,
        //   color: 0.72,
        //   bodyPosition: {
        //     region: "lower_back",
        //     x: -0.04,
        //     y: -0.1,
        //     z: -0.12
        //   }
        // },
        // {
        //   id: 7,
        //   image: "src/assets/images/image7.png",
        //   location: "Chest",
        //   probability: 0.88,
        //   skinClass: "Melanocytic",
        //   recommendedAction: "Urgent Review",
        //   dimensions: "4.9mm x 4.2mm",
        //   asymmetry: 0.84,
        //   border: 0.81,
        //   color: 0.89,
        //   bodyPosition: {
        //     region: "chest",
        //     x: 0.05,
        //     y: 0.35,
        //     z: 0.12
        //   }
        // },
        // {
        //   id: 8,
        //   image: "src/assets/images/image8.png",
        //   location: "Upper Back",
        //   probability: 0.56,
        //   skinClass: "Non-melanocytic",
        //   recommendedAction: "Monitor",
        //   dimensions: "3.4mm x 2.9mm",
        //   asymmetry: 0.52,
        //   border: 0.49,
        //   color: 0.58,
        //   bodyPosition: {
        //     region: "upper_back",
        //     x: 0.09,
        //     y: 0.28,
        //     z: -0.12
        //   }
        // },
        // {
        //   id: 9,
        //   image: "src/assets/images/image9.png",
        //   location: "Lower Back",
        //   probability: 0.78,
        //   skinClass: "Melanocytic",
        //   recommendedAction: "Review",
        //   dimensions: "4.1mm x 3.5mm",
        //   asymmetry: 0.73,
        //   border: 0.75,
        //   color: 0.81,
        //   bodyPosition: {
        //     region: "lower_back",
        //     x: 0.03,
        //     y: -0.05,
        //     z: -0.12
        //   }
        // }
      // ];

    //   // Initialize THREE.js
    //   initThreeJs();
    // });

    // THREE.js initialization
    const initThreeJs = () => {
      if (!canvasRef.value) return;

      // Setup scene
      const sceneObj = new THREE.Scene();
      sceneObj.background = new THREE.Color(0xf0f0f0);
      scene.value = sceneObj;

      // Setup camera
      const cameraObj = new THREE.PerspectiveCamera(
          35,
          canvasRef.value.clientWidth / canvasRef.value.clientHeight,
          0.1,
          1000
      );
      cameraObj.position.set(0, 0.8, 3);
      camera.value = cameraObj;

      // Setup renderer
      const rendererObj = new THREE.WebGLRenderer({
        canvas: canvasRef.value,
        antialias: true
      });
      rendererObj.setSize(canvasRef.value.clientWidth, canvasRef.value.clientHeight);
      rendererObj.setPixelRatio(window.devicePixelRatio);
      renderer.value = rendererObj;

      // Setup controls
      const controlsObj = new OrbitControls(cameraObj, rendererObj.domElement);
      controlsObj.enableDamping = true;
      controlsObj.dampingFactor = 0.05;
      controlsObj.minDistance = 1;
      controlsObj.maxDistance = 5;
      controls.value = controlsObj;

      // Add lights
      const ambientLight = new THREE.AmbientLight(0x404040, 2);
      sceneObj.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(1, 1, 1);
      sceneObj.add(directionalLight);

      // Load human model
      const loader = new FBXLoader();
      loader.load('src/assets/human_body.fbx', (object) => {
        // Adjust scale and position
        object.scale.set(0.007, 0.007, 0.007);
        object.position.set(0, -0.5, 0);

        // Apply default material
        object.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.material = new THREE.MeshPhongMaterial({
              color: 0xdddddd,
              transparent: true,
              opacity: 0.9,
              side: THREE.DoubleSide
            });
          }
        });

        sceneObj.add(object);
        model.value = object;

        // Reset camera
        cameraObj.position.set(0, 0.8, 3);
        cameraObj.lookAt(0, 0, 0);
        controlsObj.update();

        // Create highlight markers for lesions
        lesions.value.forEach((lesion) => {
          const { x, y, z } = lesion.bodyPosition;

          // Create sphere to represent lesion
          const geometry = new THREE.SphereGeometry(0.03, 32, 32);
          const material = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.9,
            visible: false
          });

          const mesh = new THREE.Mesh(geometry, material);
          mesh.position.set(x, y, z);

          // Add point light for glow effect
          const pointLight = new THREE.PointLight(0xff0000, 1, 0.2);
          pointLight.position.copy(mesh.position);
          pointLight.visible = false;
          mesh.userData.pointLight = pointLight;
          sceneObj.add(pointLight);

          sceneObj.add(mesh);

          // Store reference to highlight mesh
          highlightMeshes.value[lesion.id] = mesh;
        });
      });

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate);

        if (controls.value) {
          controls.value.update();
        }

        if (renderer.value && camera.value) {
          renderer.value.render(sceneObj, camera.value);
        }

        // Handle pulse animation for selected lesion
        if (selectedLesion.value && highlightMeshes.value[selectedLesion.value.id]) {
          const highlightMesh = highlightMeshes.value[selectedLesion.value.id];
          const scale = 1 + 0.5 * Math.sin(Date.now() * 0.005);
          highlightMesh.scale.set(scale, scale, scale);

          if (highlightMesh.material instanceof THREE.MeshBasicMaterial) {
            const intensity = 0.7 + 0.3 * Math.sin(Date.now() * 0.01);
            highlightMesh.material.color.setRGB(1, intensity * 0.3, intensity * 0.3);
          }
        }
      };

      animate();

      // Handle window resize
      const handleResize = () => {
        if (!canvasRef.value || !camera.value || !renderer.value) return;

        const width = canvasRef.value.clientWidth;
        const height = canvasRef.value.clientHeight;

        camera.value.aspect = width / height;
        camera.value.updateProjectionMatrix();

        renderer.value.setSize(width, height);
      };

      window.addEventListener('resize', handleResize);

      // Cleanup function will be handled by Vue's onUnmounted hook
    };

    // Watch for selected lesion changes
    watch(selectedLesion, (newVal) => {
      // Hide all highlights
      Object.values(highlightMeshes.value).forEach(mesh => {
        mesh.visible = false;
        if (mesh.userData.pointLight) {
          mesh.userData.pointLight.visible = false;
        }
      });

      // Show selected lesion highlight
      if (newVal && highlightMeshes.value[newVal.id]) {
        const highlightMesh = highlightMeshes.value[newVal.id];
        highlightMesh.visible = true;

        // Show point light
        if (highlightMesh.userData.pointLight) {
          highlightMesh.userData.pointLight.visible = true;
        }

        // Auto-focus on selected lesion
        if (model.value && camera.value && controls.value) {
          // Get lesion position
          const lesionPosition = new THREE.Vector3().copy(highlightMesh.position);

          // Rotate model based on lesion position
          if (newVal.bodyPosition.z < 0) {
            // Show back
            gsap.to(model.value.rotation, {
              y: 0,
              duration: 1,
              ease: "power2.inOut"
            });
          } else {
            // Show front
            gsap.to(model.value.rotation, {
              y: 0,
              duration: 1,
              ease: "power2.inOut"
            });
          }

          // Move camera to focus on lesion
          const targetY = lesionPosition.y;
          const targetZ = newVal.bodyPosition.z < 0 ? -2.5 : 2.5;

          gsap.to(camera.value.position, {
            x: lesionPosition.x * 0.5,
            y: targetY,
            z: targetZ,
            duration: 1.5,
            ease: "power2.inOut",
            onUpdate: () => {
              camera.value?.lookAt(
                  lesionPosition.x,
                  lesionPosition.y,
                  lesionPosition.z
              );
              controls.value?.update();
            }
          });
        }
      }
    });

    return {
      // State
      lesions,
      riskThresholdValue,
      riskThreshold,
      selectedLesion,
      filters,
      showRightPanel,
      enlargedImage,
      isUploadModalOpen,
      canvasRef,

      // Computed
      highRiskLesions,
      visibleLesions,

      // Methods
      handleLesionClick,
      handleNewLesion,
      resetModelView,
      parseFloat
    };
  }
};
</script>

<style>
/* Import CDN version of Tailwind CSS */
@import 'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css';

/* Component-specific styles can be added here */
</style>