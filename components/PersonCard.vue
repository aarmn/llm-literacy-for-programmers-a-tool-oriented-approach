<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps({
  people: {
    type: Array,
    required: true,
    default: () => []
  },
  title: {
    type: String,
    default: ''
  }
})

const currentIndex = ref(0)

const next = () => {
  if (currentIndex.value < props.people.length - 1) {
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

// Preset colors for platforms
const platformColors = {
  youtube: '#FF0000',
  twitter: '#1DA1F2',
  linkedin: '#0077B5',
  github: '#333333',
  arxiv: '#B31B1B',
  website: '#4285F4',
  blog: '#FF6600',
  default: '#5d8392'
}

const getPlatformColor = (platform) => {
  return platformColors[platform.toLowerCase()] || platformColors.default
}
</script>

<template>
  <div class="person-card">
    <h3 v-if="title" class="text-xl font-bold mb-2 text-left">{{ title }}</h3>
    
    <div class="relative rounded-lg p-6 shadow-lg min-h-[200px]"
         style="background-color: #5d8392; color: white;">
      
      <!-- Platform ribbon -->
      <div v-if="people[currentIndex].platforms && people[currentIndex].platforms.length > 0" 
           class="absolute -top-2 -right-2 px-2 py-1 rounded-full text-xs font-bold shadow-md"
           :style="{ backgroundColor: getPlatformColor(people[currentIndex].platforms[0].type) }">
        {{ people[currentIndex].platforms[0].type }}
      </div>
      
      <!-- Card counter -->
      <div class="absolute top-2 right-2 text-sm opacity-70">
        {{ currentIndex + 1 }} / {{ people.length }}
      </div>
      
      <!-- Card content -->
      <div class="card-content text-left flex" @click="next">
        <transition name="fade" mode="out-in">
          <div :key="currentIndex" class="py-2 flex w-full">
            <!-- Profile picture -->
            <div class="mr-4 flex-shrink-0">
              <div class="w-20 h-20 rounded-full overflow-hidden bg-gray-200 flex items-center justify-center">
                <img v-if="people[currentIndex].image" :src="people[currentIndex].image" class="w-full h-full object-cover" :alt="people[currentIndex].name">
                <div v-else class="text-gray-500 text-2xl font-bold">
                  {{ people[currentIndex].name ? people[currentIndex].name.charAt(0) : '?' }}
                </div>
              </div>
            </div>
            
            <!-- Person info -->
            <div class="flex-grow">
              <div class="font-bold text-xl mb-2">
                {{ people[currentIndex].name }}
              </div>
              <div class="mb-2">
                {{ people[currentIndex].description }}
              </div>
              
              <!-- Platforms -->
              <div v-if="people[currentIndex].platforms" class="mt-4">
                <div v-for="(platform, idx) in people[currentIndex].platforms" :key="idx" 
                     class="inline-block mr-2 mb-2 px-2 py-1 rounded text-xs font-bold"
                     :style="{ backgroundColor: getPlatformColor(platform.type) }">
                  {{ platform.type }}: {{ platform.handle }}
                </div>
              </div>
            </div>
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
          :disabled="currentIndex === people.length - 1"
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