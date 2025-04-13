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
          <button
              class="px-4 py-2 bg-blue-100 text-blue-800 rounded hover:bg-blue-200"
              @click="startAnalysis"
          >
            Start Analysis
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
            {'Head & Neck', 'Left Arm', 'Left Leg', 'Right Arm', 'Right Leg', 'Torso Back', 'Torso Front', 'Unknown'}
            <option value="all">All Regions</option>
            <option value="Head & Neck">Head & Neck</option>
            <option value="Left Arm">Left Arm</option>
            <option value="Left Leg">Left Leg</option>
            <option value="Right Arm">Right Arm</option>
            <option value="Right Leg">Right Leg</option>
            <option value="Torso Back">Torso Back</option>
            <option value="Torso Front">Torso Front</option>
            <option value="Unknown">Unknown</option>
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
              <p class="font-bold">Patient: {{ selectedLesion.patientFolderName }}</p>
              <p class="font-bold">Scan Time: {{ selectedLesion.scanTime }}</p>
              <p class="text-red-600 font-bold">
                <!--                Risk: {{ // (selectedLesion.probability * 100).toFixed(1) }}%-->
                {{ selectedLesion.probability * 100 >= 10000 ? 'Waiting for Analysis' : 'Risk: '+(selectedLesion.probability * 100).toFixed(1) + '%' }}
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
          <div class="grid gap-3" :class="showRightPanel ? 'grid-cols-5' : 'grid-cols-6'">
            <div
                v-for="lesion in visibleLesions"
                :key="lesion.id"
                class="p-2 rounded-lg border cursor-pointer transition-colors bg-gray-50 hover:shadow-lg"
                :class="{ 'ring-2 ring-blue-500': selectedLesion?.id === lesion.id }"
                @click="handleLesionClick(lesion)"
            >
              <div class="flex flex-col items-center">
                <div class="w-full aspect-square overflow-hidden rounded">
                  <img
                      :src="lesion.image"
                      :alt="`Lesion ${lesion.id}`"
                      class="object-cover w-full h-full rounded cursor-pointer"
                  />
                </div>
                <div class="w-full text-center mt-2">
                  <span
                      class="px-2 py-1 rounded text-xs font-medium"
                      :class="lesion.probability >= riskThreshold && lesion.probability * 100 < 10000
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'"
                  >
                    {{ lesion.probability * 100 >= 10000 ? 'Pending' : (lesion.probability * 100).toFixed(1) + '%' }}
                  </span>
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
                <div class="text-right">{{ selectedLesion.asymmetry !== -1 ? (selectedLesion.asymmetry * 100).toFixed(1) + '' : 'N/A' }}</div>
                <div>Border:</div>
                <div class="text-right">{{ selectedLesion.border !== -1 ? (selectedLesion.border * 100).toFixed(1) + '' : 'N/A' }}</div>
                <div>Color Variation:</div>
                <div class="text-right">{{ selectedLesion.color !== -1 ? (selectedLesion.color * 100).toFixed(1) + '' : 'N/A' }}</div>
                <div>Dimensions:</div>
                <div class="text-right">{{ selectedLesion.dimensions !== -1? (selectedLesion.dimensions * 100).toFixed(1) + '' : 'N/A'}}</div>
              </div>
            </div>

            <div class="bg-white p-4 rounded shadow">
              <h3 class="font-medium mb-2">Analysis Result</h3>
              <div class="grid grid-cols-2 gap-2 text-sm">
                <div>ud_scores_image:</div>
                <div class="text-right">{{ selectedLesion.ud_scores_image !== -1 ? (selectedLesion.ud_scores_image).toFixed(2) + '' : 'N/A' }}</div>
                <div>ud_scores_tabular:</div>
                <div class="text-right">{{ selectedLesion.ud_scores_tabular !== -1 ? (selectedLesion.ud_scores_tabular ).toFixed(2) + '' : 'N/A' }}</div>
                <div>ud_scores_imageTabular:</div>
                <div class="text-right">{{ selectedLesion.ud_scores_imageTabular !== -1 ? (selectedLesion.ud_scores_imageTabular ).toFixed(2) + '' : 'N/A' }}</div>
                <div>risk_score:</div>
                <div class="text-right">{{ selectedLesion.risk_score !== -1? (selectedLesion.risk_score).toFixed(2) + '' : 'N/A'}}</div>
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
                    <li v-if="selectedLesion.dimensions > 4">
                      Large diameter/size
                    </li>
                  </ul>
                </div>
              </div>
            </div>

            <div class="bg-white p-4 rounded shadow mt-4">
              <h3 class="font-medium mb-2">Selection Criteria</h3>
              <div class="grid grid-cols-2 gap-2 text-sm">
                <div>ID:</div>
                <div class="text-right">{{ selectedLesion.uuid || 'N/A' }}</div>

                <div>Age:</div>
                <div class="text-right">{{ selectedLesion.patientInfo.age != null ? selectedLesion.patientInfo.age : 'N/A' }}</div>

                <div>Gender:</div>
                <div class="text-right">{{ selectedLesion.patientInfo.gender || 'N/A' }}</div>

                <div>Location:</div>
                <div class="text-right">{{ selectedLesion.location || 'N/A' }}</div>

                <div>A:</div>
                <div class="text-right">{{ selectedLesion.A != null ? selectedLesion.A.toFixed(2) : 'N/A' }}</div>

                <div>Aext:</div>
                <div class="text-right">{{ selectedLesion.Aext != null ? selectedLesion.Aext.toFixed(2) : 'N/A' }}</div>

                <div>B:</div>
                <div class="text-right">{{ selectedLesion.B != null ? selectedLesion.B.toFixed(2) : 'N/A' }}</div>

                <div>Bext:</div>
                <div class="text-right">{{ selectedLesion.Bext != null ? selectedLesion.Bext.toFixed(2) : 'N/A' }}</div>

                <div>C:</div>
                <div class="text-right">{{ selectedLesion.C != null ? selectedLesion.C.toFixed(2) : 'N/A' }}</div>

                <div>Cext:</div>
                <div class="text-right">{{ selectedLesion.Cext != null ? selectedLesion.Cext.toFixed(2) : 'N/A' }}</div>

                <div>H:</div>
                <div class="text-right">{{ selectedLesion.H != null ? selectedLesion.H.toFixed(2) : 'N/A' }}</div>

                <div>Hext:</div>
                <div class="text-right">{{ selectedLesion.Hext != null ? selectedLesion.Hext.toFixed(2) : 'N/A' }}</div>

                <div>L:</div>
                <div class="text-right">{{ selectedLesion.L != null ? selectedLesion.L.toFixed(2) : 'N/A' }}</div>

                <div>Lext:</div>
                <div class="text-right">{{ selectedLesion.Lext != null ? selectedLesion.Lext.toFixed(2) : 'N/A' }}</div>

                <div>Area (mm²):</div>
                <div class="text-right">{{ selectedLesion.areaMM2 != null ? selectedLesion.areaMM2.toFixed(2) : 'N/A' }}</div>

                <div>Area/Perimeter Ratio:</div>
                <div class="text-right">{{ selectedLesion.area_perim_ratio != null ? selectedLesion.area_perim_ratio.toFixed(2) : 'N/A' }}</div>

                <div>Colour Std Mean:</div>
                <div class="text-right">{{ selectedLesion.color_std_mean != null ? selectedLesion.color_std_mean.toFixed(2) : 'N/A' }}</div>

                <div>Delta A:</div>
                <div class="text-right">{{ selectedLesion.deltaA != null ? selectedLesion.deltaA.toFixed(2) : 'N/A' }}</div>

                <div>Delta B:</div>
                <div class="text-right">{{ selectedLesion.deltaB != null ? selectedLesion.deltaB.toFixed(2) : 'N/A' }}</div>

                <div>Delta L:</div>
                <div class="text-right">{{ selectedLesion.deltaL != null ? selectedLesion.deltaL.toFixed(2) : 'N/A' }}</div>

                <div>Delta LB:</div>
                <div class="text-right">{{ selectedLesion.deltaLB != null ? selectedLesion.deltaLB.toFixed(2) : 'N/A' }}</div>

                <div>Delta LB Norm:</div>
                <div class="text-right">{{ selectedLesion.deltaLBnorm != null ? selectedLesion.deltaLBnorm.toFixed(2) : 'N/A' }}</div>

                <div>DNN Lesion Confidence (%):</div>
                <div class="text-right">{{ selectedLesion.dnn_lesion_confidence != null ? selectedLesion.dnn_lesion_confidence.toFixed(2) + '%' : 'N/A' }}</div>

                <div>Nevi Confidence (%):</div>
                <div class="text-right">{{ selectedLesion.nevi_confidence != null ? selectedLesion.nevi_confidence.toFixed(2) + '%' : 'N/A' }}</div>

                <div>Eccentricity:</div>
                <div class="text-right">{{ selectedLesion.eccentricity != null ? selectedLesion.eccentricity.toFixed(2) : 'N/A' }}</div>

                <div>Major Axis (mm):</div>
                <div class="text-right">{{ selectedLesion.majorAxisMM != null ? selectedLesion.majorAxisMM.toFixed(2) : 'N/A' }}</div>

                <div>Minor Axis (mm):</div>
                <div class="text-right">{{ selectedLesion.minorAxisMM != null ? selectedLesion.minorAxisMM.toFixed(2) : 'N/A' }}</div>

                <div>Perimeter (mm):</div>
                <div class="text-right">{{ selectedLesion.perimeterMM != null ? selectedLesion.perimeterMM.toFixed(2) : 'N/A' }}</div>

                <div>Normalised Border:</div>
                <div class="text-right">{{ selectedLesion.norm_border != null ? selectedLesion.norm_border.toFixed(2) : 'N/A' }}</div>

                <div>Normalised Colour:</div>
                <div class="text-right">{{ selectedLesion.norm_color != null ? selectedLesion.norm_color.toFixed(2) : 'N/A' }}</div>

                <div>Radial Colour Std Max:</div>
                <div class="text-right">{{ selectedLesion.radial_color_std_max != null ? selectedLesion.radial_color_std_max.toFixed(2) : 'N/A' }}</div>

                <div>Std L:</div>
                <div class="text-right">{{ selectedLesion.stdL != null ? selectedLesion.stdL.toFixed(2) : 'N/A' }}</div>

                <div>Std L Ext:</div>
                <div class="text-right">{{ selectedLesion.stdLExt != null ? selectedLesion.stdLExt.toFixed(2) : 'N/A' }}</div>

                <div>Symmetry (2 Axes):</div>
                <div class="text-right">{{ selectedLesion.symm_2axis != null ? selectedLesion.symm_2axis.toFixed(2) : 'N/A' }}</div>

                <div>Symmetry Angle (°):</div>
                <div class="text-right">{{ selectedLesion.symm_2axis_angle != null ? selectedLesion.symm_2axis_angle.toFixed(2) : 'N/A' }}</div>
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
              <button
                  class="flex-1 px-4 py-2 bg-green-100 text-green-800 rounded hover:bg-green-200"
                  @click="analyzeSingleLesion"
              >
                Analysis
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
    <!-- Analysis Progress Modal -->
    <AnalysisProgressModal
        :is-open="isAnalysisModalOpen"
        :overall-progress="analysisOverallProgress"
        :status="analysisStatus"
        :steps="analysisSteps"
        :allow-close="allowCloseAnalysisModal"
        :allow-cancel="!isAnalysisCompleted"
        :is-completed="isAnalysisCompleted"
        @close="isAnalysisModalOpen = false"
        @cancel="cancelAnalysis"
    />
  </div>
</template>

<script>
import { ipcApiRoute } from '@/api';
import { ipc } from '@/utils/ipcRenderer';
import { ref, reactive, computed, onMounted, watch } from 'vue';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import gsap from 'gsap';
import UploadScanModal from './UploadScanModal.vue';
import AnalysisProgressModal from './AnalysisProgressModal.vue';
import axios from 'axios';




export default {
  name: 'MelanomaScreening',
  components: {
    UploadScanModal,
    AnalysisProgressModal
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


    // New state for analysis progress
    const isAnalysisModalOpen = ref(false);
    const analysisOverallProgress = ref(0);
    const analysisStatus = ref('Preparing analysis...');
    const analysisSteps = ref([
      {
        name: 'Server Connection',
        progress: 0,
        status: 'pending',
        message: 'Waiting to start...'
      },
      {
        name: 'Feature Extraction',
        progress: 0,
        status: 'pending',
        message: 'Waiting to start...'
      },
      {
        name: 'ABCD Analysis',
        progress: 0,
        status: 'pending',
        message: 'Waiting to start...'
      },
      {
        name: 'Risk Assessment',
        progress: 0,
        status: 'pending',
        message: 'Waiting to start...'
      }
    ]);
    const allowCloseAnalysisModal = ref(false);
    const isAnalysisCompleted = ref(false);

    // New state for tracking individual lesion progress
    const currentLesion = ref(null);
    const totalLesions = ref(null);
    const currentLesionId = ref(null);

    const analyzeSingleLesion = () => {
      if (!selectedLesion.value) {
        // No lesion selected
        alert('Please select a lesion to analyze');
        return;
      }
      // Call the main analysis method with just the selected lesion
      simulateAnalysisProcess(selectedLesion.value);
    };
      // Simulated analysis function
    const startAnalysis = () => {
      // Only start if we have lesions to analyze
      if (lesions.value.length === 0) {
        alert('No lesions to analyze. Please upload a scan first.');
        return;
      }
      // // Call the main simulation process with no specific lesion (analyze all)
      simulateAnalysisProcess();
    };

    const simulateAnalysisProcess = async (singleLesion = null) => {
      try {
        // Set the lesions to analyze - either all or just one
        const lesionsToAnalyze = singleLesion ? [singleLesion] : lesions.value;

        // Only start if we have lesions to analyze
        if (lesionsToAnalyze.length === 0) {
          throw new Error('No lesions to analyze. Please upload a scan first.');
        }

        // Reset analysis state
        analysisOverallProgress.value = 0;
        analysisStatus.value = singleLesion ?
            'Preparing analysis for selected lesion...' :
            'Preparing analysis...';

        analysisSteps.value.forEach(step => {
          step.progress = 0;
          step.status = 'pending';
          step.message = 'Waiting to start...'
        });
        allowCloseAnalysisModal.value = false;
        isAnalysisCompleted.value = false;

        // Initialize tracking variables for current lesion
        currentLesion.value = 0;
        totalLesions.value = lesionsToAnalyze.length;
        currentLesionId.value = null;

        // Open the modal
        isAnalysisModalOpen.value = true;

        // Step 1: Testing server connection with /hello endpoint
        analysisStatus.value = 'Testing connection to server...';
        const step1 = analysisSteps.value[0];
        step1.status = 'in-progress';
        step1.message = 'Connecting to analysis server...';
        step1.progress = 30;

        // Test connection to the server using IPC to call /hello endpoint
        try {
          console.log('Testing connection to server with /hello endpoint');

          // Use IPC to call the hello endpoint
          const response = await ipc.invoke(ipcApiRoute.melanoma.helloServer);

          if (!response || response.status !== 'success') {
            throw new Error('Server did not return a success response');
          }

          console.log('Server connection test successful:', response);

          // Update step progress
          step1.progress = 100;
          step1.status = 'completed';
          step1.message = 'Server connection established successfully';

        } catch (error) {
          console.error('Server connection test failed:', error);
          step1.status = 'error';
          step1.message = `Failed to connect to server: ${error.message || 'Unknown error'}`;

          // Don't continue with analysis if server connection failed
          analysisStatus.value = 'Analysis failed: Cannot connect to server';
          allowCloseAnalysisModal.value = true;

          // Mark step 4 as not started
          const step4 = analysisSteps.value[3];
          step4.status = 'pending';
          step4.message = 'Server connection failed';

          return;
        }

        // Brief pause after connection test
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Create a reference to the risk assessment step
        const step4 = analysisSteps.value[3];
        step4.status = 'pending';
        step4.message = 'Waiting for connection test to complete...';

        // Process each lesion one at a time
        const processedLesions = [];

        for (let i = 0; i < lesionsToAnalyze.length; i++) {
          // Update current lesion tracking
          currentLesion.value = i + 1;
          currentLesionId.value = lesionsToAnalyze[i].id;

          // Update overall progress based on current lesion
          analysisOverallProgress.value = Math.round((i / lesionsToAnalyze.length) * 100);

          // Update status with current lesion info
          analysisStatus.value = `Analyzing lesion ${i + 1} of ${lesionsToAnalyze.length}`;

          // Update risk assessment step
          step4.status = 'in-progress';
          step4.message = `Evaluating lesion ${i + 1} of ${lesionsToAnalyze.length}`;
          step4.progress = Math.round((i / lesionsToAnalyze.length) * 100);

          try {
            // Prepare single lesion data object
            const lesion = lesionsToAnalyze[i];
            const lesionData = {
              id: lesion.id,
              image: typeof lesion.image === 'string' ? lesion.image : null,
              gender: lesion.patientInfo?.gender || '',
              age: lesion.patientInfo?.age || 0,
              A: lesion.A || 0,
              Aext: lesion.Aext || 0,
              B: lesion.B || 0,
              Bext: lesion.Bext || 0,
              C: lesion.C || 0,
              Cext: lesion.Cext || 0,
              H: lesion.H || 0,
              Hext: lesion.Hext || 0,
              L: lesion.L || 0,
              Lext: lesion.Lext || 0,
              areaMM2: lesion.areaMM2 || 0,
              area_perim_ratio: lesion.area_perim_ratio || 0,
              color_std_mean: lesion.color_std_mean || 0,
              deltaA: lesion.deltaA || 0,
              deltaB: lesion.deltaB || 0,
              deltaL: lesion.deltaL || 0,
              deltaLB: lesion.deltaLB || 0,
              deltaLBnorm: lesion.deltaLBnorm || 0,
              dnn_lesion_confidence: lesion.dnn_lesion_confidence || 0,
              eccentricity: lesion.eccentricity || 0,
              location_simple: lesion.location || '',
              majorAxisMM: lesion.majorAxisMM || 0,
              minorAxisMM: lesion.minorAxisMM || 0,
              nevi_confidence: lesion.nevi_confidence || 0,
              norm_border: lesion.norm_border || 0,
              norm_color: lesion.norm_color || 0,
              perimeterMM: lesion.perimeterMM || 0,
              radial_color_std_max: lesion.radial_color_std_max || 0,
              stdL: lesion.stdL || 0,
              stdLExt: lesion.stdLExt || 0,
              symm_2axis: lesion.symm_2axis || 0,
              symm_2axis_angle: lesion.symm_2axis_angle || 0,
            };

            console.log(`Analyzing lesion ${i + 1} of ${lesionsToAnalyze.length}: ${lesion.id}`);

            // Call Python backend
            const response = await ipc.invoke(ipcApiRoute.melanoma.analyzeLesions, {
              lesions: [lesionData]
            });

            console.log(`Lesion ${i + 1} analysis response:`, response);

            // Add processed lesion to results array
            if (response && response.lesions && response.lesions.length > 0) {
              processedLesions.push(response.lesions[0]);

              // Simulate a brief pause between lesions
              await new Promise(resolve => setTimeout(resolve, 500));
            }
          } catch (error) {
            console.error(`Error processing lesion ${i + 1}:`, error);
            analysisStatus.value = `Error analyzing lesion ${i + 1}: ${error.message}`;
            step4.message = `Error analyzing lesion ${i + 1}: ${error.message}`;

            // Brief pause to show error message
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }

        // Analysis complete
        analysisStatus.value = singleLesion ?
            'Analysis of selected lesion complete!' :
            'Analysis complete!';
        analysisOverallProgress.value = 100;

        // Update step 4 to completed
        step4.status = 'completed';
        step4.progress = 100;
        step4.message = `All ${lesionsToAnalyze.length} lesions analyzed successfully`;

        allowCloseAnalysisModal.value = true;
        isAnalysisCompleted.value = true;
        currentLesion.value = null; // Clear current lesion indicator
        currentLesionId.value = null;

        // Update lesions with analysis results
        if (processedLesions.length > 0) {
          updateLesionsWithAnalysisResults(processedLesions);
        }

      } catch (error) {
        console.error('Error in analysis process:', error);
        analysisStatus.value = 'Analysis failed: ' + error.message;
        allowCloseAnalysisModal.value = true;

        // Mark step 4 as error
        const step4 = analysisSteps.value[3];
        step4.status = 'error';
        step4.message = 'Analysis failed: ' + error.message;
      }
    };


    // Update to accept external analysis results
    const updateLesionsWithAnalysisResults = (analysisResults) => {
      lesions.value = lesions.value.map(lesion => {
        // 找到匹配的分析结果
        const result = analysisResults.find(r => r.id === lesion.id);

        if (result) {
          console.log('Analysis Results:', result.ud_scores_image, result.risk_score, result.uuid);

          // 合并新的分析结果到现有的病变对象
          // 如果分析结果中某个值为null或undefined，则设置为-1，这样模板中的条件就能正确工作
          return {
            ...lesion,
            // 使用新的输出参数
            ud_scores_image: result.ud_scores_image ?? -1, // 使用空值合并运算符
            ud_scores_tabular: result.ud_scores_tabular ?? -1,
            ud_scores_imageTabular: result.ud_scores_imageTabular ?? -1,
            risk_score: result.risk_score ?? -1,

            // 把risk_score映射到probability以保持UI兼容性
            probability: result.risk_score || 0,

            // 保留现有的ABCD分析数据或使用默认值
            asymmetry: lesion.asymmetry ?? -1,
            border: lesion.border ?? -1,
            color: lesion.color ?? -1,
            dimensions: lesion.dimensions ?? -1
          };
        }
        return lesion;
      });

      // 如果当前选中的病变在分析结果中，则更新selectedLesion引用
      // 这确保了右侧面板会刷新
      if (selectedLesion.value) {
        const updatedSelectedLesion = lesions.value.find(l => l.id === selectedLesion.value.id);
        if (updatedSelectedLesion) {
          selectedLesion.value = updatedSelectedLesion;
        }
      }
    };
    const cancelAnalysis = () => {
      // In a real application, you would stop the analysis process here
      isAnalysisModalOpen.value = false;
    };

    // Computed properties
    const processedLesions = computed(() => {
      return [...lesions.value]
          .sort((a, b) => b.probability - a.probability)
          .filter(lesion => {
            if (filters.bodyRegion !== 'all' && lesion.location !== filters.bodyRegion) return false;
            return true;
            // const width = lesion.dimensions;
            // return width >= filters.minSize && width <= filters.maxSize;
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
          z:0,
          duration: 0.3
        });

        if (camera.value && controls.value) {
          // Reset camera position
          gsap.to(camera.value.position, {
            x: 0,
            y: 0.8,
            z: 3,
            duration: 0.3,
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

      // Add top-down directional light
      const topLight = new THREE.AmbientLight(0xffffff, 1.2);
      topLight.position.set(0, 3, 0); // Position directly above
      topLight.lookAt(0, 0, 0);
      sceneObj.add(topLight);

      // Add subtle fill light from front for better visibility
      const frontFillLight = new THREE.AmbientLight(0xffffff, 1);
      frontFillLight.position.set(0, 0, 1);
      sceneObj.add(frontFillLight);

      // Add subtle back light for depth
      const backLight = new THREE.AmbientLight(0xffffff, 1);
      backLight.position.set(0, 0, -1);
      sceneObj.add(backLight);

      // Define a mapping between location names and bone names
      // const locationToBoneMap = {
      //   'Head & Neck': ['mixamorig1Head', 'mixamorig1Neck', 'mixamorig1HeadTop_End'],
      //   'Left Arm': ['mixamorig1LeftArm', 'mixamorig1LeftForeArm', 'mixamorig1LeftHand', 'mixamorig1LeftShoulder'],
      //   'Right Arm': ['mixamorig1RightArm', 'mixamorig1RightForeArm', 'mixamorig1RightHand', 'mixamorig1RightShoulder'],
      //   'Left Leg': ['mixamorig1LeftUpLeg', 'mixamorig1LeftLeg', 'mixamorig1LeftFoot', 'mixamorig1LeftToeBase'],
      //   'Right Leg': ['mixamorig1RightUpLeg', 'mixamorig1RightLeg', 'mixamorig1RightFoot', 'mixamorig1RightToeBase'],
      //   'Torso Front': ['mixamorig1Spine', 'mixamorig1Spine1', 'mixamorig1Spine2', 'mixamorig1Hips'],
      //   'Torso Back': ['mixamorig1Spine', 'mixamorig1Spine1', 'mixamorig1Spine2', 'mixamorig1Hips'],
      //   'Unknown': []
      // };

      const locationToBoneMap = {
        'Head & Neck': ['mixamorig1Neck'],
        'Left Arm': ['mixamorig1LeftForeArm'],
        'Right Arm': ['mixamorig1RightForeArm'],
        'Left Leg': ['mixamorig1LeftLeg'],
        'Right Leg': ['mixamorig1RightLeg'],
        'Torso Front': ['mixamorig1FontTwo'],
        'Torso Back': ['mixamorig1Hips'],
        'Unknown': []
      };
      // Store references to bones
      const bonesMap = ref({});

      // Store references to original materials to revert when needed
      const originalMaterials = ref({});

      // Load human model
      const loader = new FBXLoader();
      loader.load('models/human_body.fbx', (object) => {
        // Adjust scale and position
        object.scale.set(0.007, 0.007, 0.007);
        object.position.set(0, -0.5, 0);

        // Apply default material and store bone references
        object.traverse((child) => {
          // Debug: Log all object names and types for analysis
          // console.log(`Object found: ${child.name}, Type: ${child.type}, isBone: ${child.isBone ? 'Yes' : 'No'}`);

          if (child.isMesh) {
            child.material.side = THREE.DoubleSide;
            child.material.transparent = true;
            child.material.needsUpdate = true;
            child.metalness = 0;

            // Store the original material for this mesh
            originalMaterials.value[child.id] = child.material.clone();

            // Store meshes by name for region highlighting
            // This helps when bones aren't directly accessible
            if (child.name.includes('Arm') || child.name.includes('arm') ||
                child.name.includes('Hand') || child.name.includes('hand') ||
                child.name.includes('Leg') || child.name.includes('leg') ||
                child.name.includes('Foot') || child.name.includes('foot') ||
                child.name.includes('Head') || child.name.includes('head') ||
                child.name.includes('Torso') || child.name.includes('torso') ||
                child.name.includes('Spine') || child.name.includes('spine')) {
              // console.log(`Found relevant mesh: ${child.name}`);
            }
          }

          // Try different ways to detect bones in the model
          // Some FBX models use isBone flag, others use different properties
          if (child.isBone ||
              child.type === 'Bone' ||
              (child.name && (
                  child.name.includes('mixamorig') ||
                  child.name.includes('Bone') ||
                  child.name.includes('bone') ||
                  child.name.includes('Arm') ||
                  child.name.includes('Leg') ||
                  child.name.includes('Head') ||
                  child.name.includes('Spine')
              ))) {
            const boneName = child.name;
            bonesMap.value[boneName] = child;
            // console.log(`Stored bone: ${boneName}`);
          }
        });

        sceneObj.add(object);
        model.value = object;

        // Reset camera
        cameraObj.position.set(0, 0.8, 3);
        cameraObj.lookAt(0, 0, 0);
        controlsObj.update();

        // Create highlight markers for lesions
        createLesionMarkers();

      });

      // Function to create lesion markers
      const createLesionMarkers = () => {
        lesions.value.forEach((lesion) => {
          // Get bone positions based on location
          const bonesToHighlight = locationToBoneMap[lesion.location] || [];

          if (bonesToHighlight.length > 0) {
            // Find the related bones and create a marker at each bone
            bonesToHighlight.forEach(boneName => {
              const bone = bonesMap.value[boneName];

              if (bone) {
                // Get the world position of the bone
                const position = new THREE.Vector3();
                bone.getWorldPosition(position);

                // Create sphere to represent lesion
                const geometry = new THREE.SphereGeometry(0.03, 32, 32);
                const material = new THREE.MeshBasicMaterial({
                  color: 0xff0000,
                  transparent: true,
                  opacity: 0.9,
                  visible: false
                });

                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.copy(position);

                // Store bone reference in mesh userData
                mesh.userData.bone = bone;
                mesh.userData.boneName = boneName;

                // Add point light for glow effect
                const pointLight = new THREE.PointLight(0xff0000, 1, 0.2);
                pointLight.position.copy(mesh.position);
                pointLight.visible = false;
                mesh.userData.pointLight = pointLight;
                sceneObj.add(pointLight);

                sceneObj.add(mesh);

                // Store reference to highlight mesh
                if (!highlightMeshes.value[lesion.id]) {
                  highlightMeshes.value[lesion.id] = [];
                }
                highlightMeshes.value[lesion.id].push(mesh);
              }
            });
          } else {
            // Fallback to the old positioning method if no bone mapping exists
            const { x, y, z } = lesion.bodyPosition || { x: 0, y: 0, z: 0 };

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
            if (!highlightMeshes.value[lesion.id]) {
              highlightMeshes.value[lesion.id] = [];
            }
            highlightMeshes.value[lesion.id].push(mesh);
          }
        });
      };
      // 在highlightBodyRegion函数开始时彻底清除之前的所有标记
      const clearAllHighlights = () => {
        // 移除所有已存在的标记
        Object.keys(highlightMeshes.value).forEach(lesionId => {
          const markers = highlightMeshes.value[lesionId];
          markers.forEach(mesh => {
            if (mesh.parent) {
              mesh.parent.remove(mesh);
            }
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) mesh.material.dispose();

            // 清理相关的点光源
            if (mesh.userData.pointLight && mesh.userData.pointLight.parent) {
              mesh.userData.pointLight.parent.remove(mesh.userData.pointLight);
            }
          });
        });

        // 重置高亮网格对象
        highlightMeshes.value = {};

        // 重置所有网格材质到原始状态
        model.value.traverse((child) => {
          if (child.isMesh && originalMaterials.value[child.id]) {
            child.material = originalMaterials.value[child.id].clone();
            child.material.needsUpdate = true;
          }
        });
      };

      // Function to highlight the body region corresponding to a lesion
      const highlightBodyRegion = (lesion) => {
        clearAllHighlights();
        // Reset all meshes to original materials first
        model.value.traverse((child) => {
          if (child.isMesh && originalMaterials.value[child.id]) {
            child.material = originalMaterials.value[child.id].clone();
            child.material.needsUpdate = true;
          }
        });

        if (!lesion) return;

        console.log(`Attempting to highlight region for location: ${lesion.location}`);

        // Get the bones related to this location
        const bonesToHighlight = locationToBoneMap[lesion.location] || [];

        // Log all bone names for debugging
        console.log("Available bones in model:");
        Object.keys(bonesMap.value).forEach(boneName => {
          const bone = bonesMap.value[boneName];
          const position = new THREE.Vector3();
          bone.getWorldPosition(position);
          console.log(`Bone: ${boneName}, Position: x=${position.x.toFixed(2)}, y=${position.y.toFixed(2)}, z=${position.z.toFixed(2)}`);
        });

        // Log the bones we're looking for
        console.log("Bones to highlight for this location:", bonesToHighlight);

        // Perform only exact bone name matching
        let foundExactMatch = false;

        if (bonesToHighlight.length > 0) {
          // Create a set of bone names for faster lookup
          const boneNameSet = new Set(bonesToHighlight);

          // Find if any of our target bones exist in the model
          for (const boneName of bonesToHighlight) {
            const bone = bonesMap.value[boneName];

            if (bone) {
              console.log(`Found exact bone match: ${boneName}`);
              foundExactMatch = true;

              // Find associated meshes by traversing the model
              model.value.traverse((child) => {
                if (child.isMesh) {
                  // Only match meshes that are directly connected to this bone
                  if (child.skeleton && child.skeleton.bones.some(b => b.name === boneName)) {
                    console.log(`Highlighting mesh: ${child.name} (connected to bone: ${boneName})`);
                    applyHighlightMaterial(child);
                  }
                }
              });

              // Create a marker at the bone position for visibility
              const position = new THREE.Vector3();
              bone.getWorldPosition(position);

              // Create a sphere to highlight the region
              const geometry = new THREE.SphereGeometry(0.08, 32, 32);
              const material = new THREE.MeshBasicMaterial({
                color: 0xff0000,
                transparent: true,
                opacity: 0.5
              });

              const marker = new THREE.Mesh(geometry, material);
              marker.position.copy(position);

              // Add to scene
              scene.value.add(marker);

              // Store for cleanup later
              if (!highlightMeshes.value[lesion.id]) {
                highlightMeshes.value[lesion.id] = [];
              }
              highlightMeshes.value[lesion.id].push(marker);
            }
          }
        }

        if (!foundExactMatch) {
          console.log(`No exact bone matches found for location: ${lesion.location}. No highlighting applied.`);

          // Add a visualization of the location map
          console.log("Location to Bone Map:");
          for (const [location, bones] of Object.entries(locationToBoneMap)) {
            console.log(`${location}: ${bones.join(', ')}`);
          }
        }
      };

      // Helper function to apply highlight material to a mesh
      const applyHighlightMaterial = (mesh) => {
        // Create a highlighted material
        const highlightMaterial = new THREE.MeshStandardMaterial({
          color: 0xffdddd,          // Light red tint
          emissive: 0xff0000,       // Red glow
          emissiveIntensity: 0.3,   // Subtle glow
          metalness: 0.0,
          roughness: 0.6,
          transparent: true,
          opacity: 0.9,
          side: THREE.DoubleSide
        });

        // Copy maps from original material if they exist
        if (originalMaterials.value[mesh.id]) {
          const origMat = originalMaterials.value[mesh.id];
          if (origMat.map) highlightMaterial.map = origMat.map;
          if (origMat.normalMap) highlightMaterial.normalMap = origMat.normalMap;
        }

        // Apply the highlight material
        mesh.material = highlightMaterial;
        mesh.material.needsUpdate = true;
      };

      // Store the highlightBodyRegion function in a reactive ref for use in watchers
      highlightBodyRegionFunc.value = highlightBodyRegion;

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
          const lesionMarkers = highlightMeshes.value[selectedLesion.value.id];

          // Pulse animation for all markers of the selected lesion
          lesionMarkers.forEach(marker => {
            const scale = 1 + 0.5 * Math.sin(Date.now() * 0.005);
            marker.scale.set(scale, scale, scale);

            if (marker.material instanceof THREE.MeshBasicMaterial) {
              const intensity = 0.7 + 0.3 * Math.sin(Date.now() * 0.01);
              marker.material.color.setRGB(1, intensity * 0.3, intensity * 0.3);
            }

            // Update point light intensity as well
            if (marker.userData.pointLight) {
              marker.userData.pointLight.intensity = 1 + 0.5 * Math.sin(Date.now() * 0.005);
            }
          });
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

// Add this ref to your setup function
    const highlightBodyRegionFunc = ref(null);

// Then update your watch() function for selectedLesion
    watch(selectedLesion, (newVal) => {
      // Hide all highlights
      Object.keys(highlightMeshes.value).forEach(lesionId => {
        const markers = highlightMeshes.value[lesionId];
        markers.forEach(mesh => {
          mesh.visible = false;
          if (mesh.userData.pointLight) {
            mesh.userData.pointLight.visible = false;
          }
        });
      });

      // Highlight the body region of the selected lesion
      if (highlightBodyRegionFunc.value) {
        highlightBodyRegionFunc.value(newVal);
      }

      // Show selected lesion highlights
      if (newVal && highlightMeshes.value[newVal.id]) {
        const markers = highlightMeshes.value[newVal.id];

        // Show all markers for this lesion
        markers.forEach(marker => {
          marker.visible = true;

          // Show point light
          if (marker.userData.pointLight) {
            marker.userData.pointLight.visible = true;
          }
        });

        // If there's at least one marker, use it for camera positioning
        if (markers.length > 0) {
          const primaryMarker = markers[0];

          // Auto-focus on selected lesion
          if (model.value && camera.value && controls.value) {
            // Get lesion position
            const lesionPosition = new THREE.Vector3().copy(primaryMarker.position);

            // Properly determine if we need front or back view based on the lesion location
            let needsBackView = false;

            if (newVal.location === 'Torso Back') {
              needsBackView = true;
            } else if (newVal.location === 'Torso Front') {
              needsBackView = false;
            } else if (newVal.bodyPosition && newVal.bodyPosition.z < 0) {
              // If we have explicit position data showing it's on the back
              needsBackView = false;
            }

            // Rotate model based on whether we need front or back view
            const targetRotationY = needsBackView ? Math.PI : 0; // 180 degrees for back view

            gsap.to(model.value.rotation, {
              y: targetRotationY,
              duration: 0.3,
              ease: "power2.inOut"
            });

            const targetY = lesionPosition.y;
            // Always position camera in front of the model (positive Z)
            // The model rotation will determine if we see front or back
            const targetZ = 2.5;

            gsap.to(camera.value.position, {
              x: lesionPosition.x * 0.5,
              y: targetY,
              z: targetZ,
              duration: 0.3,
              ease: "power2.inOut",
              onUpdate: () => {
                camera.value?.lookAt(
                    lesionPosition.x,
                    lesionPosition.y,
                    0 // Look at the center of the model
                );
                controls.value?.update();
              }
            });
          }
        }
      }
    });

    return {
      // State
      analyzeSingleLesion,
      lesions,
      riskThresholdValue,
      riskThreshold,
      selectedLesion,
      filters,
      showRightPanel,
      enlargedImage,
      isUploadModalOpen,
      canvasRef,

      // Analysis state
      isAnalysisModalOpen,
      analysisOverallProgress,
      analysisStatus,
      analysisSteps,
      allowCloseAnalysisModal,
      isAnalysisCompleted,
      currentLesion,      // New
      totalLesions,       // New
      currentLesionId,    // New
      startAnalysis,
      cancelAnalysis,

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