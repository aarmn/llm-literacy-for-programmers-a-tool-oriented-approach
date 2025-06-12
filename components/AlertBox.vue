<template>
  <div :class="alertClasses" class="alert-box">
    <div class="alert-icon">
      <component :is="iconComponent" />
    </div>
    <div class="alert-content">
      <h4 v-if="title" class="alert-title">{{ title }}</h4>
      <div class="alert-message">
        <slot></slot>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'

// Simple icon components
const InfoIcon = {
  template: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
  </svg>`
}

const WarningIcon = {
  template: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
  </svg>`
}

const NoteIcon = {
  template: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
  </svg>`
}

const TipIcon = {
  template: `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M9,21C9,22.1 9.9,23 11,23H13C14.1,23 15,22.1 15,21V20H9V21M12,2A7,7 0 0,0 5,9C5,11.38 6.19,13.47 8,14.74V17A1,1 0 0,0 9,18H15A1,1 0 0,0 16,17V14.74C17.81,13.47 19,11.38 19,9A7,7 0 0,0 12,2Z"/>
  </svg>`
}

export default {
  name: 'AlertBox',
  components: {
    InfoIcon,
    WarningIcon,
    NoteIcon,
    TipIcon
  },
  props: {
    type: {
      type: String,
      default: 'info',
      validator: (value) => ['info', 'warning', 'note', 'tip', 'danger', 'success'].includes(value)
    },
    title: {
      type: String,
      default: ''
    }
  },
  setup(props) {
    const alertClasses = computed(() => {
      return `alert alert-${props.type}`
    })

    const iconComponent = computed(() => {
      const iconMap = {
        info: 'InfoIcon',
        warning: 'WarningIcon',
        note: 'NoteIcon',
        tip: 'TipIcon',
        danger: 'WarningIcon',
        success: 'InfoIcon'
      }
      return iconMap[props.type] || 'InfoIcon'
    })

    return {
      alertClasses,
      iconComponent
    }
  }
}
</script>

<style scoped>
.alert-box {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  border-radius: 8px;
  border-left: 4px solid;
  margin: 16px 0;
  backdrop-filter: blur(10px);
}

.alert-icon {
  flex-shrink: 0;
  margin-top: 2px;
}

.alert-content {
  flex: 1;
}

.alert-title {
  margin: 0 0 8px 0;
  font-weight: 600;
  font-size: 16px;
}

.alert-message {
  margin: 0;
  line-height: 1.5;
}

/* Type-specific styles */
.alert-info {
  background: rgba(59, 130, 246, 0.1);
  border-left-color: #3b82f6;
  color: #1e40af;
}

.alert-info .alert-icon {
  color: #3b82f6;
}

.alert-warning {
  background: rgba(245, 158, 11, 0.1);
  border-left-color: #f59e0b;
  color: #92400e;
}

.alert-warning .alert-icon {
  color: #f59e0b;
}

.alert-note {
  background: rgba(99, 102, 241, 0.1);
  border-left-color: #6366f1;
  color: #4338ca;
}

.alert-note .alert-icon {
  color: #6366f1;
}

.alert-tip {
  background: rgba(16, 185, 129, 0.1);
  border-left-color: #10b981;
  color: #065f46;
}

.alert-tip .alert-icon {
  color: #10b981;
}

.alert-danger {
  background: rgba(239, 68, 68, 0.1);
  border-left-color: #ef4444;
  color: #991b1b;
}

.alert-danger .alert-icon {
  color: #ef4444;
}

.alert-success {
  background: rgba(34, 197, 94, 0.1);
  border-left-color: #22c55e;
  color: #166534;
}

.alert-success .alert-icon {
  color: #22c55e;
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
  .alert-info {
    background: rgba(59, 130, 246, 0.15);
    color: #93c5fd;
  }
  
  .alert-warning {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
  }
  
  .alert-note {
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
  }
  
  .alert-tip {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
  }
  
  .alert-danger {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
  }
  
  .alert-success {
    background: rgba(34, 197, 94, 0.15);
    color: #86efac;
  }
}
</style>