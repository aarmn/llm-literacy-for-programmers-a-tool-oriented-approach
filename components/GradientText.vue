<script setup lang="ts">
const props = defineProps<{
  text?: string
}>()
</script>

<template>
  <div class="gradient-text-wrapper">
    <span class="gradient-text">
      <slot>{{ text }}</slot>
    </span>
  </div>
</template>

<style scoped>
.gradient-text-wrapper {
  display: inline-block;
}

.gradient-text {
  background: linear-gradient(
    45deg,
    #ff6b9d,
    #c06bff,
    #6bb5ff,
    #6bffc0,
    #ffd96b,
    #ff6b9d
  );
  background-size: 400% 400%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: gradient-shift 8s ease infinite;
  font-weight: 600;
}

@keyframes gradient-shift {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}
</style>
