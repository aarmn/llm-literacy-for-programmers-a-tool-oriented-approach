<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps({
  items: {
    type: Array,
    required: true,
    default: () => []
  },
  title: {
    type: String,
    default: ''
  },
  showExamples: {
    type: Boolean,
    default: true
  },
  highlightIndex: {
    type: Number,
    default: -1
  }
})

const currentIndex = ref(0)

const isHighlighted = computed(() => {
  return currentIndex.value === props.highlightIndex
})

const next = () => {
  if (currentIndex.value < props.items.length - 1) {
    currentIndex.value++
  }
}

const prev = () => {
  if (currentIndex.value > 0) {
    currentIndex.value--
  }
}

const reset = () => {
  currentIndex.value = 0
}

const goToHighlighted = () => {
  if (props.highlightIndex >= 0 && props.highlightIndex < props.items.length) {
    currentIndex.value = props.highlightIndex
  }
}
</script>

<template>
  <div class="card-deck">
    <h3 v-if="title" class="text-xl font-bold mb-2 text-left">{{ title }}</h3>
    
    <div class="relative rounded-lg p-6 shadow-lg min-h-[200px]"
         :class="{ 'ring-4 ring-yellow-400 dark:ring-yellow-500': isHighlighted }"
         style="background-color: #5d8392; color: white;">
      <!-- Highlight indicator -->
      <div v-if="isHighlighted" class="absolute -top-2 -right-2 bg-yellow-400 text-black px-2 py-1 rounded-full text-xs font-bold shadow-md">
        Highlighted
      </div>
      
      <!-- Card counter -->
      <div class="absolute top-2 right-2 text-sm opacity-70" :class="{ 'mr-20': isHighlighted }">
        {{ currentIndex + 1 }} / {{ items.length }}
      </div>
      
      <!-- Card content -->
      <div class="card-content text-left" @click="next">
        <transition name="fade" mode="out-in">
          <div :key="currentIndex" class="py-2">
            <slot 
              :item="items[currentIndex]" 
              :index="currentIndex" 
              :showExamples="showExamples"
              :highlighted="isHighlighted"
            >
              <div v-if="typeof items[currentIndex] === 'string'">
                {{ items[currentIndex] }}
              </div>
              <div v-else>
                <div class="font-bold text-xl mb-2" :class="{ 'text-yellow-300': isHighlighted }">
                  {{ items[currentIndex].title }}
                </div>
                <div v-if="showExamples && items[currentIndex].examples">
                  <div class="mb-2 mt-4">
                    <div class="text-red-300 font-medium">Bad:</div>
                    <div class="bg-gray-700/30 p-2 rounded">
                      {{ items[currentIndex].examples.bad }}
                    </div>
                  </div>
                  <div class="mb-2">
                    <div class="text-green-300 font-medium">Good:</div>
                    <div class="bg-gray-700/30 p-2 rounded">
                      {{ items[currentIndex].examples.good }}
                    </div>
                  </div>
                </div>
                <div v-else-if="items[currentIndex].content">
                  {{ items[currentIndex].content }}
                </div>
              </div>
            </slot>
          </div>
        </transition>
      </div>
      
      <!-- Navigation buttons -->
      <div class="flex justify-between mt-4">
        <button 
          @click.stop="prev" 
          class="p-2 bg-white text-[#5d8392] rounded-full hover:bg-gray-200 disabled:opacity-50 flex items-center justify-center"
          :disabled="currentIndex === 0"
          aria-label="Previous"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd" />
          </svg>
        </button>
        <div class="flex gap-2">
          <button 
            @click.stop="reset" 
            class="p-2 bg-white text-[#5d8392] rounded-full hover:bg-gray-200 flex items-center justify-center"
            aria-label="Reset"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd" />
            </svg>
          </button>
        </div>
        <button 
          @click.stop="next" 
          class="p-2 bg-white text-[#5d8392] rounded-full hover:bg-gray-200 disabled:opacity-50 flex items-center justify-center"
          :disabled="currentIndex === items.length - 1"
          aria-label="Next"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.card-content {
  cursor: pointer;
  min-height: 120px;
  display: flex;
  align-items: flex-start;
  justify-content: flex-start;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>



