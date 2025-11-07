<script setup>
import { ref, onMounted } from 'vue'

const americanLabs = [
  {
    name: "ChatGPT",
    color: "#4a9eff",
    region: "american",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1762508698/pplx_project_search_images/d9f20e5eea431c68b44a27c9c2c4603480d5a802.png",
    backedBy: "OpenAI",
    backingLogo: "https://pplx-res.cloudinary.com/image/upload/v1762508698/pplx_project_search_images/d9f20e5eea431c68b44a27c9c2c4603480d5a802.png"
  },
  {
    name: "Google DeepMind",
    color: "#4a9eff",
    region: "american",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1754743738/pplx_project_search_images/67580f662af73026d54fe0872aea32ec707b9d3d.png",
    backedBy: "Google",
    backingLogo: "https://crystalpng.com/wp-content/uploads/2025/05/google-logo.png"
  },
  {
    name: "Microsoft Copilot",
    color: "#2e7fd1",
    region: "american",
    logo: "https://registry.npmmirror.com/@lobehub/icons-static-png/1.74.0/files/dark/copilot-color.png",
    backedBy: "Microsoft",
    backingLogo: "https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/Microsoft_logo.svg/1024px-Microsoft_logo.svg.png?20210729021049"
  },
  {
    name: "Meta AI",
    color: "#4a9eff",
    region: "american",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1756113120/pplx_project_search_images/746fe05c71c461aa613685d94addf6845b2ea309.png",
    backedBy: "Meta",
    backingLogo: "https://pngimg.com/uploads/meta/meta_PNG4.png"
  },
  {
    name: "xAI",
    color: "#2e7fd1",
    region: "american",
    logo: "https://registry.npmmirror.com/@lobehub/icons-static-png/1.74.0/files/dark/xai.png",
    backedBy: "Elon Musk",
    backingLogo: "https://pplx-res.cloudinary.com/image/upload/v1756421348/pplx_project_search_images/650cde82a8e5c180eb793b9a6aee59159e123f6c.png"
  },
  {
    name: "Claude",
    color: "#4a9eff",
    region: "american",
    logo: "https://registry.npmmirror.com/@lobehub/icons-static-png/1.74.0/files/dark/claude-color.png",
    backedBy: "Anthropic",
    backingLogo: "https://registry.npmmirror.com/@lobehub/icons-static-png/1.74.0/files/dark/anthropic.png"
  },
  {
    name: "AI2",
    color: "#4a9eff",
    region: "american",
    logo: "https://image2url.com/images/1762509181328-95de1232-30aa-4bba-bce0-c61be2a60c5f.png",
    backedBy: "Independent",
    backingLogo: ""
  },
  {
    name: "Perplexity",
    color: "#4a9eff",
    region: "american",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1754731592/pplx_project_search_images/0680dbae45d6e292bf6eb3ca24d102ef3d6a3835.png",
    backedBy: "Independent",
    backingLogo: ""
  },
  {
    name: "Manus AI",
    color: "#4a9eff",
    region: "american",
    logo: "https://ai-bot.cn/wp-content/uploads/2025/03/Manus-logo.png",
    backedBy: "Independent",
    backingLogo: ""
  }
]

const europeanLabs = [
  {
    name: "Mistral AI",
    color: "#999999",
    region: "european",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1755001854/pplx_project_search_images/d8b1e069293cff1fab1088d4ee17e6f08309a9e6.png",
    backedBy: "Independent",
    backingLogo: ""
  }
]

const internationalLabs = [
  {
    name: "Inclusion AI",
    color: "#ff4757",
    region: "international",
    logo: "https://cdn-avatars.huggingface.co/v1/production/uploads/662e1f9da266499277937d33/fyKuazRifqiaIO34xrhhm.jpeg",
    backedBy: "Ant Group",
    backingLogo: "https://pplx-res.cloudinary.com/image/upload/v1762476362/pplx_project_search_images/86003706e044641190bd862e4753ad8dd55d624b.png"
  },
  {
    name: "Qwen",
    color: "#ff6b6b",
    region: "international",
    logo: "https://cdn.brandfetch.io/idoXsIxRHZ/w/150/h/150/theme/light/logo.png?c=1bxid64Mup7aczewSAYMX&t=1762207416205",
    backedBy: "Alibaba",
    backingLogo: "https://alibabamadeeasy.com.au/wp-content/uploads/2020/07/alibaba-cloud-logo.png"
  },
  {
    name: "DeepSeek",
    color: "#ff4757",
    region: "international",
    logo: "https://freepnglogo.com/images/all_img/deep-seek-logo-whale-1ced.png",
    backedBy: "High-Flyer",
    backingLogo: "https://www.marefa.org/w/images/b/bd/High-Flyer.png"
  },
  {
    name: "Z.ai",
    color: "#ff6b6b",
    region: "international",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1762476375/pplx_project_search_images/6479736d105969cdb31a61d18bd0f6a4c34ee9ad.png",
    backedBy: "Independent",
    backingLogo: ""
  },
  {
    name: "Moonshot AI",
    color: "#ff4757",
    region: "international",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1762476858/pplx_project_search_images/a21008e3eb42351d480ce0dc0b218e3741a2761f.png",
    backedBy: "Independent",
    backingLogo: ""
  },
  {
    name: "ERNIE (Baidu)",
    color: "#ff6b6b",
    region: "international",
    logo: "https://registry.npmmirror.com/@lobehub/icons-static-png/1.74.0/files/dark/wenxin-color.png",
    backedBy: "Baidu",
    backingLogo: "https://companieslogo.com/img/orig/BIDU-4322f316.png?t=1654487760"
  },
  {
    name: "ByteDance",
    color: "#ff4757",
    region: "international",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1762476116/pplx_project_search_images/2df8acadd95a7b8b23c4ec61ffb965f2c9ea8a19.png",
    backedBy: "Independent",
    backingLogo: ""
  },
  {
    name: "Tencent (Hunyuan)",
    color: "#ff6b6b",
    region: "international",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1757386727/pplx_project_search_images/ece5db8debf1988f25bdba67b3af03d02df02f90.png",
    backedBy: "Tencent",
    backingLogo: "https://pplx-res.cloudinary.com/image/upload/v1762475394/pplx_project_search_images/5a50d793bc9dec3b482950a3377ac9385d3f3fdc.png"
  },
  {
    name: "MiniMax",
    color: "#ff4757",
    region: "international",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1757386740/pplx_project_search_images/182bb59e3078bc906831f5f461dfda78a2f44a82.png",
    backedBy: "Independent",
    backingLogo: ""
  },
  {
    name: "StepFun",
    color: "#ff6b6b",
    region: "international",
    logo: "https://pplx-res.cloudinary.com/image/upload/v1757386740/pplx_project_search_images/963a886cfacfcc30a6f8f9461454e17e3a28557d.png",
    backedBy: "Independent",
    backingLogo: ""
  }
]

const aiLabs = [...americanLabs, ...europeanLabs, ...internationalLabs]

function throttle(callback, limit) {
  let waiting = false
  return function() {
    if (!waiting) {
      callback.apply(this, arguments)
      waiting = true
      setTimeout(() => {
        waiting = false
      }, limit)
    }
  }
}

onMounted(() => {
  const cards = document.querySelectorAll('.ai-card')
  
  cards.forEach(card => {
    card.addEventListener('pointermove', throttle(function(e) {
      const rect = card.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const centerX = rect.width / 2
      const centerY = rect.height / 2
      const tiltX = ((y - centerY) / centerY) * 8
      const tiltY = ((centerX - x) / centerX) * 8
      card.style.setProperty('--tilt-x', `${tiltX.toFixed(2)}deg`)
      card.style.setProperty('--tilt-y', `${tiltY.toFixed(2)}deg`)
      card.style.willChange = 'transform'
    }, 18))
    
    card.addEventListener('pointerenter', () => {
      card.style.setProperty('--tilt-x', '0deg')
      card.style.setProperty('--tilt-y', '0deg')
      card.style.willChange = 'transform'
    })
    
    card.addEventListener('pointerleave', () => {
      card.style.setProperty('--tilt-x', '0deg')
      card.style.setProperty('--tilt-y', '0deg')
      setTimeout(() => {
        card.style.willChange = 'auto'
      }, 400)
    })
  })
})
</script>

<template>
  <div class="ai-labs-container">
    <div class="ai-labs-grid">
      <div
        v-for="(lab, index) in aiLabs"
        :key="index"
        :class="['ai-card', `ai-card--${lab.region}`]"
        :style="{ animationDelay: `${index * 0.05}s` }"
      >
        <div class="ai-glow" :style="{ boxShadow: `0 0 44px 5px ${lab.color}` }"></div>
        <div class="ai-card-inner">
          <div class="ai-logo-container">
            <img :src="lab.logo" :alt="lab.name" class="ai-logo ai-main-logo">
            <template v-if="lab.backingLogo">
              <div class="ai-backing-separator"></div>
              <img :src="lab.backingLogo" :alt="lab.backedBy" class="ai-backing-logo">
            </template>
          </div>
          <div class="ai-company-name">
            {{ lab.name }}
            <template v-if="lab.backingLogo"> &nbsp;→&nbsp; {{ lab.backedBy }}</template>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ai-labs-container {
  width: 100%;
  height: 100%;
  padding: 1.5rem;
  overflow-y: auto;
}

.ai-labs-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  perspective: 1000px;
  width: 100%;
  margin: 0 auto;
}

@media (min-width: 900px) {
  .ai-labs-grid {
    grid-template-columns: repeat(4, 1fr);
  }
}

.ai-card {
  position: relative;
  background: rgba(255, 255, 255, 0.03);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  cursor: pointer;
  transition: box-shadow 0.35s cubic-bezier(0.18, 0.85, 0.23, 1), 
              border-color 0.35s cubic-bezier(0.18, 0.85, 0.23, 1), 
              transform 0.36s cubic-bezier(0.18, 0.85, 0.23, 1);
  transform-style: preserve-3d;
  overflow: hidden;
  opacity: 0;
  animation: fadeInUp 0.6s ease-out forwards;
  aspect-ratio: 1 / 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.ai-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), rgba(255, 255, 255, 0.08), transparent 60%);
  opacity: 0;
  transition: opacity 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
}

.ai-card:hover::before {
  opacity: 1;
}

.ai-card:hover {
  transform: scale(1.055) rotateX(var(--tilt-x, 0deg)) rotateY(var(--tilt-y, 0deg));
  border-color: rgba(255, 255, 255, 0.18);
  z-index: 15;
  box-shadow: 0 6px 48px 3px rgba(255, 255, 255, 0.045), 0 0 90px 14px var(--glow-hue, transparent);
}

.ai-card--american {
  border-color: rgba(74, 158, 255, 0.15);
}

.ai-card--american:hover {
  border-color: rgba(74, 158, 255, 0.42);
  --glow-hue: rgba(74, 158, 255, 0.28);
}

.ai-card--american::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
  background: radial-gradient(circle at center, rgba(74, 158, 255, 0.1), transparent 70%);
  opacity: 0;
  transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
}

.ai-card--american:hover::after {
  opacity: 1;
}

.ai-card--european {
  border-color: rgba(153, 153, 153, 0.15);
}

.ai-card--european:hover {
  border-color: rgba(153, 153, 153, 0.42);
  --glow-hue: rgba(136, 153, 170, 0.24);
}

.ai-card--european::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
  background: radial-gradient(circle at center, rgba(153, 153, 153, 0.1), transparent 70%);
  opacity: 0;
  transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
}

.ai-card--european:hover::after {
  opacity: 1;
}

.ai-card--international {
  border-color: rgba(255, 107, 107, 0.15);
}

.ai-card--international:hover {
  border-color: rgba(255, 107, 107, 0.42);
  --glow-hue: rgba(255, 71, 87, 0.28);
}

.ai-card--international::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 12px;
  background: radial-gradient(circle at center, rgba(255, 107, 107, 0.1), transparent 70%);
  opacity: 0;
  transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
}

.ai-card--international:hover::after {
  opacity: 1;
}

.ai-card-inner {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
  padding-bottom: 3rem;
  position: relative;
  z-index: 2;
}

.ai-logo-container {
  width: 100%;
  height: 100%;
  display: grid;
  grid-template-columns: 1fr 0fr 0fr;
  align-items: center;
  justify-items: center;
  position: relative;
  transition: all 0.35s cubic-bezier(0.18, 0.85, 0.23, 1);
}

.ai-card:hover .ai-logo-container {
  transform: scale(0.95);
}

.ai-card:has(.ai-backing-logo[src]) .ai-logo-container {
  grid-template-columns: 1fr 0fr 0fr;
}

.ai-card:has(.ai-backing-logo[src]):hover .ai-logo-container {
  grid-template-columns: 45fr 1fr 45fr;
  gap: 0.5rem;
}

.ai-logo {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  filter: brightness(0.97) drop-shadow(0 4px 10px rgba(0, 0, 0, 0.16));
  transition: filter 0.36s cubic-bezier(0.18, 0.85, 0.23, 1);
  display: block;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
}

.ai-card:hover .ai-logo {
  filter: brightness(1.11) drop-shadow(0 6px 24px rgba(0, 0, 0, 0.17));
}

.ai-backing-logo {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  filter: brightness(0.92) drop-shadow(0 2px 8px rgba(0, 0, 0, 0.12));
  transition: filter 0.36s cubic-bezier(0.18, 0.85, 0.23, 1);
  opacity: 0;
  display: block;
  image-rendering: -webkit-optimize-contrast;
  image-rendering: crisp-edges;
}

.ai-card:hover .ai-backing-logo {
  opacity: 1;
  filter: brightness(1.08) drop-shadow(0 4px 16px rgba(0, 0, 0, 0.14));
}

.ai-backing-separator {
  width: 1px;
  height: 40%;
  background: rgba(255, 255, 255, 0.15);
  opacity: 0;
  transition: opacity 0.35s cubic-bezier(0.18, 0.85, 0.23, 1);
  border-radius: 2px;
  justify-self: center;
}

.ai-card:hover .ai-backing-separator {
  opacity: 1;
}

.ai-company-name {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%) translateY(8px);
  padding: 7px 16px;
  background: rgba(14, 20, 36, 0.92);
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  text-align: center;
  letter-spacing: 0.02em;
  opacity: 0;
  transition: opacity 0.27s cubic-bezier(0.18, 0.85, 0.23, 1), 
              transform 0.33s cubic-bezier(0.18, 0.85, 0.23, 1);
  pointer-events: none;
  border-radius: 8px;
  white-space: nowrap;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
  backdrop-filter: blur(11px);
  z-index: 12;
  max-width: 90%;
  overflow: hidden;
  text-overflow: ellipsis;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: geometricPrecision;
  backface-visibility: hidden;
  transform-style: preserve-3d;
}

.ai-card:hover .ai-company-name {
  opacity: 1;
  transform: translateX(-50%) translateY(0) translateZ(0);
}

.ai-glow {
  position: absolute;
  top: -4px;
  left: -4px;
  right: -4px;
  bottom: -4px;
  border-radius: 12px;
  opacity: 0;
  transition: opacity 0.5s cubic-bezier(0.16, 1, 0.3, 1);
  pointer-events: none;
  z-index: -1;
}

.ai-card:hover .ai-glow {
  opacity: 0.72;
  filter: blur(32px);
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
