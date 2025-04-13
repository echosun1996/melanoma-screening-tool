<template>
  <div v-if="isOpen" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
    <div class="bg-white rounded-lg shadow-xl p-6 w-full max-w-lg">
      <div class="flex justify-between items-center mb-4">
        <h2 class="text-xl font-bold">Analyzing Lesions</h2>
        <button v-if="allowClose" @click="$emit('close')" class="text-gray-600 hover:text-gray-800">
          <svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M18 6L6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>

      <div class="space-y-6">
        <!-- Main analysis progress -->
        <div>
          <div class="flex justify-between mb-1">
            <p>Overall Progress</p>
            <p>{{ overallProgress }}%</p>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-4">
            <div
                class="bg-blue-600 h-4 rounded-full transition-all duration-300 ease-in-out"
                :style="{ width: `${overallProgress}%` }"
            ></div>
          </div>
          <p class="text-sm text-gray-600 mt-1">{{ status }}</p>

          <!-- Current lesion progress indicator -->
          <div v-if="currentLesion && totalLesions" class="mt-2 bg-blue-50 border border-blue-200 rounded p-2">
            <p class="text-sm text-blue-800">
              Processing lesion {{ currentLesion }} of {{ totalLesions }}
              <span v-if="currentLesionId" class="font-mono text-xs block mt-1">ID: {{ currentLesionId }}</span>
            </p>
          </div>
        </div>

        <!-- Individual step progress - only showing first and fourth steps -->
        <div v-for="(step, index) in filteredSteps" :key="index" class="border-t pt-4">
          <div class="flex justify-between mb-1">
            <p class="flex items-center">
              <span
                  class="inline-block w-5 h-5 rounded-full mr-2 flex items-center justify-center text-xs"
                  :class="getStepStatusClass(step.status)"
              >
                <svg v-if="step.status === 'completed'" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                </svg>
                <svg v-else-if="step.status === 'error'" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                </svg>
              </span>
              {{ step.name }}
            </p>
            <p>{{ step.progress }}%</p>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-2">
            <div
                class="h-2 rounded-full transition-all duration-300 ease-in-out"
                :class="getProgressBarColor(step.status)"
                :style="{ width: `${step.progress}%` }"
            ></div>
          </div>
          <p class="text-xs text-gray-600 mt-1">{{ step.message }}</p>
        </div>
      </div>

      <div class="mt-6 flex justify-end space-x-2">
        <button
            v-if="allowCancel && !isCompleted"
            @click="$emit('cancel')"
            class="px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200"
        >
          Cancel Analysis
        </button>
        <button
            v-if="isCompleted || allowClose"
            @click="$emit('close')"
            :class="isCompleted ? 'bg-green-600 text-white hover:bg-green-700' : 'bg-gray-200 text-gray-800 hover:bg-gray-300'"
            class="px-4 py-2 rounded"
        >
          {{ isCompleted ? 'View Results' : 'Close' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "AnalysisProgressModal",
  props: {
    isOpen: {
      type: Boolean,
      default: false
    },
    overallProgress: {
      type: Number,
      default: 0
    },
    status: {
      type: String,
      default: "Preparing analysis..."
    },
    steps: {
      type: Array,
      default: () => []
    },
    allowClose: {
      type: Boolean,
      default: false
    },
    allowCancel: {
      type: Boolean,
      default: true
    },
    isCompleted: {
      type: Boolean,
      default: false
    },
    // New props for tracking individual lesion progress
    currentLesion: {
      type: Number,
      default: null
    },
    totalLesions: {
      type: Number,
      default: null
    },
    currentLesionId: {
      type: String,
      default: null
    }
  },
  computed: {
    // Filter to only show the first and fourth steps
    filteredSteps() {
      if (!this.steps || this.steps.length === 0) return [];

      const result = [];

      // Add first step (Image Pre-processing)
      if (this.steps.length >= 1) {
        result.push(this.steps[0]);
      }

      // Add fourth step (Risk Assessment)
      if (this.steps.length >= 4) {
        result.push(this.steps[3]);
      }

      return result;
    }
  },
  methods: {
    getStepStatusClass(status) {
      switch (status) {
        case 'completed':
          return 'bg-green-500 text-white';
        case 'in-progress':
          return 'bg-blue-500 text-white';
        case 'pending':
          return 'bg-gray-300 text-gray-600';
        case 'error':
          return 'bg-red-500 text-white';
        default:
          return 'bg-gray-300 text-gray-600';
      }
    },
    getProgressBarColor(status) {
      switch (status) {
        case 'completed':
          return 'bg-green-500';
        case 'in-progress':
          return 'bg-blue-500';
        case 'pending':
          return 'bg-gray-300';
        case 'error':
          return 'bg-red-500';
        default:
          return 'bg-gray-300';
      }
    }
  }
};
</script>