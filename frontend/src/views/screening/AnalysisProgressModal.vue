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
        </div>

        <!-- Individual step progress -->
        <div v-for="(step, index) in steps" :key="index" class="border-t pt-4">
          <div class="flex justify-between mb-1">
            <p class="flex items-center">
              <span
                  class="inline-block w-5 h-5 rounded-full mr-2 flex items-center justify-center text-xs"
                  :class="getStepStatusClass(step.status)"
              >
                <svg v-if="step.status === 'completed'" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
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

      <div class="mt-6 flex justify-end">
        <button
            v-if="isCompleted"
            @click="$emit('close')"
            class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
        >
          View Results
        </button>
        <button
            v-else-if="allowCancel"
            @click="$emit('cancel')"
            class="px-4 py-2 bg-red-100 text-red-800 rounded hover:bg-red-200"
        >
          Cancel Analysis
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