<script setup lang="ts">
import { ref, computed } from 'vue'

const props = defineProps<{
  initialValues?: number[]
}>()

const values = ref(props.initialValues || [0.1, 0.15, 0.2, 0.25, 0.3])
const temperature = ref(1.0)
const topP = ref(1.0)
const topK = ref(5)
const logitBias = ref<Record<number, number>>({})

const applyTemperature = (probs: number[], temp: number) => {
  if (temp === 0) return probs.map((_, i) => i === probs.indexOf(Math.max(...probs)) ? 1 : 0)
  const logits = probs.map(p => Math.log(p + 1e-10))
  const scaledLogits = logits.map(l => l / temp)
  const maxLogit = Math.max(...scaledLogits)
  const expLogits = scaledLogits.map(l => Math.exp(l - maxLogit))
  const sum = expLogits.reduce((a, b) => a + b, 0)
  return expLogits.map(e => e / sum)
}

const applyTopK = (probs: number[], k: number) => {
  const indexed = probs.map((p, i) => ({ p, i }))
  indexed.sort((a, b) => b.p - a.p)
  const result = new Array(probs.length).fill(0)
  const topKItems = indexed.slice(0, k)
  const sum = topKItems.reduce((acc, item) => acc + item.p, 0)
  topKItems.forEach(item => {
    result[item.i] = item.p / sum
  })
  return result
}

const applyTopP = (probs: number[], p: number) => {
  const indexed = probs.map((prob, i) => ({ p: prob, i }))
  indexed.sort((a, b) => b.p - a.p)
  let cumSum = 0
  const result = new Array(probs.length).fill(0)
  const selected: typeof indexed = []
  for (const item of indexed) {
    if (cumSum >= p) break
    selected.push(item)
    cumSum += item.p
  }
  const sum = selected.reduce((acc, item) => acc + item.p, 0)
  selected.forEach(item => {
    result[item.i] = item.p / sum
  })
  return result
}

const applyLogitBias = (probs: number[], bias: Record<number, number>) => {
  const logits = probs.map((p, i) => Math.log(p + 1e-10) + (bias[i] || 0))
  const maxLogit = Math.max(...logits)
  const expLogits = logits.map(l => Math.exp(l - maxLogit))
  const sum = expLogits.reduce((a, b) => a + b, 0)
  return expLogits.map(e => e / sum)
}

const processedProbs = computed(() => {
  let probs = [...values.value]
  const sum = probs.reduce((a, b) => a + b, 0)
  probs = probs.map(p => p / sum)
  probs = applyLogitBias(probs, logitBias.value)
  probs = applyTemperature(probs, temperature.value)
  probs = applyTopK(probs, topK.value)
  probs = applyTopP(probs, topP.value)
  return probs
})

const maxProb = computed(() => Math.max(...processedProbs.value))

const resetParams = () => {
  temperature.value = 1.0
  topP.value = 1.0
  topK.value = 5
  logitBias.value = {}
}
</script>

<template>
  <div class="sampling-card">
    <div class="card-header">
      <h3>Memoryless Sampling Hyperparameters</h3>
      <button @click="resetParams" class="reset-btn">Reset</button>
    </div>
    
    <div class="two-cols">
      <div class="bar-chart">
        <div v-for="(prob, index) in processedProbs" :key="index" class="bar-container">
          <div class="bar-label">Token {{ index + 1 }}</div>
          <div class="bar-wrapper">
            <div 
              class="bar"
              :style="{ 
                width: `${(prob / maxProb) * 100}%`,
                opacity: prob === 0 ? 0.1 : 1
              }"
            />
            <span class="bar-value">{{ (prob * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>

      <div class="controls">
        <div class="control-group">
          <label>
            Temperature: {{ temperature.toFixed(2) }}
            <input v-model.number="temperature" type="range" min="0" max="2" step="0.1" />
          </label>
        </div>

        <div class="control-group">
          <label>
            Top-P: {{ topP.toFixed(2) }}
            <input v-model.number="topP" type="range" min="0" max="1" step="0.05" />
          </label>
        </div>

        <div class="control-group">
          <label>
            Top-K: {{ topK }}
            <input v-model.number="topK" type="range" min="1" :max="values.length" step="1" />
          </label>
        </div>

        <div class="control-group">
          <label>Logit Bias</label>
          <div class="logit-bias-inputs">
            <input 
              v-for="(_, index) in values" 
              :key="index"
              v-model.number="logitBias[index]"
              type="number"
              :placeholder="`T${index + 1}`"
              step="0.5"
              class="bias-input"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.sampling-card {
  padding: 1rem 0;
  margin: 1rem 0;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.card-header h3 {
  margin: 0;
  font-size: 1.2rem;
}

.reset-btn {
  padding: 0.4rem 0.8rem;
  background: rgba(59, 130, 246, 0.2);
  border: 1px solid rgba(59, 130, 246, 0.4);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.reset-btn:hover {
  background: rgba(59, 130, 246, 0.3);
}

.two-cols {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.bar-chart {
  padding: 1rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
}

.bar-container {
  display: flex;
  align-items: center;
  margin-bottom: 0.8rem;
  gap: 1rem;
}

.bar-label {
  min-width: 80px;
  font-size: 0.9rem;
  opacity: 0.8;
}

.bar-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.bar {
  height: 24px;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  border-radius: 4px;
  transition: all 0.3s ease;
  min-width: 2px;
}

.bar-value {
  min-width: 50px;
  font-size: 0.85rem;
  opacity: 0.7;
}

.controls {
  display: grid;
  gap: 1.2rem;
}

.control-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  opacity: 0.9;
}

.control-group input[type="range"] {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  outline: none;
}

.control-group input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #3b82f6;
  cursor: pointer;
}

.logit-bias-inputs {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.bias-input {
  width: 60px;
  padding: 0.4rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  color: inherit;
  font-size: 0.85rem;
}

.bias-input:focus {
  outline: none;
  border-color: #3b82f6;
}
</style>
