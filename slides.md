---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: ./assets/bg.png
# some information about your slides, markdown enabled
title: LLM Literacy for Post-LLM Programmer
info: |
  ## LLM Literacy for Post-LLM Programmer
  An opinionated, practical guide to vibe-coding in a vibing world, where not leveraging AI is like choosing horseback over air travel.

  By AARMN The Limitless
# apply any unocss classes to the current slide
class: text-center
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# https://sli.dev/guide/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/guide/syntax#mdc-syntax
mdc: true
---

<!-- 
TODO:
- Licensing of these, open weight, open source, closed source, closed weight, ..., 
- Ollama, and hand on
- Open Web UI
- tangent technologies, git, docker, vector dbs, python, 
- good tooling vs common toolings
- update on techs
- update of current situation
- add quantization and LoRA
Know High-leverage tooling
key things to look for in an LLM
param count, context length, MoE, 

-->

# LLM Literacy for Post-LLM Programmer

An opinionated, practical guide to vibe-coding in a vibing world, where not leveraging AI is like choosing horseback over air travel. 

**2nd** Edition

<div class="abs-br m-6 flex gap-2">
  <a href="[Link to your socials/repo]" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---
transition: fade-out
---

# Who I am?

<div class="grid grid-cols-2 gap-8">
<div>

<!-- TODO: Add a picture of yourself or a relevant avatar -->
<img src="./assets/me.jpeg" class="rounded-full mx-auto" style="width: 200px; height: 200px; object-fit: cover;" alt="Your Name">

<div class="mt-4 text-center">
  <h3 class="text-xl">Alireza Mohammadnejad</h3>
  <p class="text-sm opacity-75">Guilan University AI Master's Student | LLM Enthusiast</p>
  <p class="text-sm opacity-60">Professional vibecoder (!) with a strong dedication toward optimization of workflows and automation</p>
</div>

</div>
<div>

Why should you listen to me (for the next two hours, anyway)?

<v-clicks>

- I used LLMs since first preview of InstructionGPT
- I've built a lot of cool stuff using LLMs.
- I use LLMs daily, and I wont tell you stop using LLMs
- I've made packages with (`pixelist`) and for (`vibecopy`) LLMs ...
- I improve my workflows constantly.
- I don't have an H100 GPU in the garage, just like you.
- My main goal: Demystify AI, Help you leverage it by "Least price" and "Highest yield"

</v-clicks>

</div>
</div>

---
layoutClass: gap-16
---

<div style="display: grid; place-items: center; height: 100%;">
  <h1>Let me Bring a News Slide from the First Edition of This Presentation</h1>
</div>

---
layoutClass: gap-16
---

<v-clicks>

# Let's Take a Look at Highlights of Announcements From the Last Week!

</v-clicks>

<v-clicks>

- <div class="flex items-center gap-2">
    <img src="./assets/DeepMind-New-Logo-Vector.svg-.png" class="w-6 h-6" />
    AlphaEvolve <span class="text-blue-400">By Google</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/google-gemini-icon.svg" class="w-6 h-6" />
    Gemini Diffusion <span class="text-blue-400">By Google</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/bagel.png" class="w-6 h-6" />
    Bagel <span class="text-blue-400">by Bytedance</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/claude-ai-icon.svg" class="w-6 h-6" />
    Claude 4 models <span class="text-blue-400">by Anthropic</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/chatgpt-icon.svg" class="w-6 h-6" />
    Codex (New) <span class="text-blue-400">by OpenAI</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/google-gemini-icon.svg" class="w-6 h-6" />
    Gemini 2.5 Flash Preview (05-20) <span class="text-blue-400">by Google</span>
  </div>

## Therefore Keep in Mind

<AlertBox type="warning">
AI never sleeps, <s>and you shouldn't either</s>. Update your high leverage tools, but focus on learning fundamentals deeply!
</AlertBox>

</v-clicks>

<!--
Mention Gemma3n as well, which we can guess from its size, but no info about it is released and its format as a new unique format, never before seen.

Mention these is two path toward it based on this info
-->

---
transition: fade-out
---

<div style="display: grid; place-items: center; height: 100%;">
  <h1>Time Flies, Huh? What a Week That Was!</h1>
</div>

---
transition: fade-out
---

<div style="display: grid; place-items: center; height: 100%;">
  <div>
    <h1>This time, We Have Less Hot News, As It's Not Announcement Season, but...
    <span v-click>AI Never Slows Down, While Announcements Might</span></h1>
  </div>
</div>


---
transition: fade-out
---

# Let's Look at Hot News of Circa Last Month

<v-clicks>

- <div class="flex items-center gap-2">
    <img src="./assets/google-gemini-icon.svg" class="w-6 h-6" />
    Gemini Pro 3.0 <span class="text-gray-400">(Codename <code>lithiumflow</code> in LMArena)</span> <span class="text-blue-400">By Google</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/google-gemini-icon.svg" class="w-6 h-6" />
    Gemini Flash 3.0 <span class="text-gray-400">(Codename <code>orionmist</code> in LMArena)</span> <span class="text-blue-400">By Google</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/DeepMind-New-Logo-Vector.svg-.png" class="w-6 h-6" />
    EmbeddingGemma <span class="text-blue-400">By Google</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/zai.svg" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">GLM 4.6</span> <span class="text-blue-400">By ZhipuAI</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/qwen.png" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">Qwen3 VL</span> <span class="text-gray-400"> (Varieties for size, quantization, thinking) </span><span class="text-blue-400">by Qwen (Alibaba Cloud) </span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/claude-ai-icon.svg" class="w-6 h-6" />
    Claude Sonnet 4.5 model <span class="text-blue-400">by Anthropic</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/claude-ai-icon.svg" class="w-6 h-6" />
    Claude Haiku 4.5 model <span class="text-blue-400">by Anthropic</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/minimax.png" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">Minimax M2</span> <span class="text-gray-400">(openweighted)</span> <span class="text-blue-400">by MiniMax</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/kimi.png" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">Kimi K2 Instruct</span> <span class="text-gray-400">(09-05)</span> <span class="text-blue-400">by Moonshot AI</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/deepseek.png" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">DeepSeek V3.2</span> <span class="text-gray-400">(Exp)</span> <span class="text-blue-400">by DeepSeekAI (HighFlyer)</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/inclusionAI.jpeg" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">Ling 1T and Ring 1T</span> <span class="text-blue-400">by InclusionAI (Ant Group)</span>
  </div>
- <div class="flex items-center gap-2">
    <img src="./assets/inclusionAI.jpeg" class="w-6 h-6" />
    <span :class="$clicks >= 13 ? 'text-red-400' : ''">LLaDA 2.0 Flash Preview</span> <span class="text-gray-400">(Diffusion base)</span><span class="text-blue-400">by InclusionAI (Ant Group)</span>
  </div>

</v-clicks>
   
<v-clicks><div></div></v-clicks>

---
transition: fade-out
---

<!-- show duality of deepseek - zai - qwen - minimax - Inclusion AI - Kimi | Google - OpenAI - Anthropic - Meta - Ai2 - NVidia -  -->

---
transition: fade-out
---

<div style="display: grid; place-items: center; height: 100%;">
  <h1>We Look Deeper to Trends in The End!</h1>
</div>

---
transition: fade-out
---

# What I want you to take away?

<v-clicks>

- **What** is an LLM? <!-- (And how does it "think"?) -->
- How a **model** is different from a **provider**?
- How to **make** an LLM? <span class="text-gray-400"> (The 10,000-foot view) </span>
- Why LLMs are not a **fad**? 
- What is **todays and probably not tomorrow** (!) SoTA? <span class="text-gray-400"> (by vibes and benchmarks) </span>
- Why LLMs **on their own** are not a **silver bullet**? <span class="text-gray-400">(ft. LLM vs Human anatomy)</span> <!-- (downsides and how to mitigate them using The tooling and human element) -->
- Which workflows and toolings elevate LLM to next-level?
- How to Ask the right questions? <span class="text-gray-400">(Art, Philosophy, Psychology, Socialogy, AI and Engineering, of Prompting)</span>
- How to run a model on your very own system? <span class="text-gray-400">(Thy shall not kneel to big brother cloud!)</span>
- LLM and Security <span class="text-gray-400">(Attack vectors and opportunities + SLMs)</span>
- Quantization <span class="text-gray-400">(What? Why? Why not? How?)</span>
- Benchmarks 
- Whats next? <span class="text-gray-400">(Focused on trend insight not *Choo Choo* hype train)</span>

</v-clicks>

<!--
This is our roadmap. Each point will be expanded.
-->

---
transition: fade-out
---


<div style="display: grid; place-items: center; height: 100%;">
  <div>
    <h1>In the Next Few Slides, We Will See:
    <br><br>
    <div v-click>Theoretical Ingredients for a Model</div></h1>
  </div>
</div>

---
transition: fade-out
---

# Architecture

<v-clicks>

- The blueprint of how the model should function
- Similar to Classes in Programming, not an instance, but a scheme
- how parts interact and learn or not from data. It's the math and code of the machinery. It describes how model learns, often by tuning some internal "knobs" (parameters) based on the data it sees.
- Tells the How

</v-clicks>

---
transition: fade-out
---

# Parameters

<v-clicks>

- The internal "knobs" the model tunes during training
- More knobs = more capacity to learn nuance 
- Insight (often inferred data) is stored in here!
- this is the largest part of the final model artifact.
- A 70B parameters architecture, therefore, has 70 Billion **learnable** numbers, which model has learnt

</v-clicks>

<br>

<v-clicks>

# Hyper‌parameters

</v-clicks>

<v-clicks>

- What is the difference?
- Why not learn these?
  - We can't, as they are fundementally different! (efficently)
    - Hyperparameter sweep! (per use-case)
  - Gives us, per-use case, degree of freedom on inference stage, without a need for data

</v-clicks>

---
transition: fade-out
---

# Training Data

<v-clicks>

- Terabytes of text and code. 
- For SoTA models, basically, As big of a chunk, they can get off of Internet
- Differs from model to model, as architecture differ
- Training data differs, in **shape**, **quality** and **variety** per **use-case** and **stage of training** the model
- The pre-training phase is currently the most data hungry part of the process, quantity over quality (in LLMs)
- Data-quality and format in later stages matter alot more (in LLMs)

</v-clicks>

---

<div style="display: grid; place-items: center; height: 100%;">
  <div>
    <h1>And to Find the Correct Parameters... <span v-click>We Need Computation!</span></h1>
  </div>
</div>

---

<div style="display: grid; place-items: center; height: 100%;">
  <div>
    <h1>Before That, 2 Critical Words: <span v-click>Inference and training</span></h1>
  </div>
</div>

---

# Physical Ingredients for a Model

<v-clicks>

- **Computation**: The raw power for calculation, in both learning and inference
  - Learning models require learning from data, before or interleaved with their inference stage
  - The inference itself requires computation
  - This ~~is~~ was the most expensive stage computationally for SoTA LLMs
  - What the is becoming was means for AI?

- **Energy**: Deeply entangled with computation
  - Training and inference at scale require significant energy resources
  - Many tech companies in the AI boom are building their own power plants due to sheer demand

- **Hardware**: AI computation requires specialized hardware
  - Often embarrassingly parallel, making GPUs far better than CPUs with current architectures

- **Experts!**

<!-- and ALL‌ THESE‌ Points tooooooo -->

</v-clicks>

<img v-click src="./assets/money.png" class="absolute" style="top: 130px; right: 10px; width: 500px; transform: rotate(+15deg);" alt="Money">

---
transition: fade-out
---

# A little analogy

<v-clicks>

### Human child

</v-clicks>

<v-clicks>

- **Architecture**: DNA, the blueprint of how the child should function, how parts interact and learn or not from data.

- **Parameters**: The internal "knobs" the child tunes during learning, from birth to adulthood. this can be their neurons connections, their muscles shape, ...

- **Training Data**: The experiences, interactions, and observations the child encounters from birth to adulthood. This data shapes the child's understanding of the world and how they interact with it.

- **Computation and Energy**: The child's brainpower and the energy it expends to learn and grow. handled by the food they eat. It's the rolling of DNA into a human being which happens since egg cells forward, powered by energy and chemical reactions.

</v-clicks>

---
transition: fade-out
---

# Commonly Used Jargon: Decoding the Lingo

<v-clicks>

- **Language Model (LM):** Any system (often AI-based) that models (understands and/or generates using probabilities) human language. Anything from autocompletes from a decade ago, till modern giants.

- **Large Language Model (LLM):** An LM, but **BIG**.
  - **Large** in terms of:
    1. **Parameters**: more than ~~100 million~~ ~~1 billion~~ 10 billion <!-- main ingredient of the Large badge -->
    2. Training Data (have terabytes to petabytes of training data)
    3. Generalized capabilities (across tasks, domains, languages)

- **Multimodal / VLM (Vision Language Model):** LMs that can process and understand information from multiple types of data, not just text. Common combo is text + images.
  - *Examples:* GPT-4o (closed-weight), Llava (open-weight).



</v-clicks>

--- 

# Commonly Used Jargon: Decoding the Lingo

<v-clicks>

- **SoTA (State-of-the-Art)**: The best-performing model/technique for a specific task *right now*.

- **Cutting-edge/Bleeding-edge**: Very new, promising, but maybe not fully proven or widely adopted SoTA.

- **Diffusion**: An architecture which is commonly used in image generation but recently is being explored for text, its main ability, is speed and non‑autoregressive generation, making it perfect for blank filling, and probably, style transfer of text. 

- When I mention **transformer model** or **diffusion model**, I mean a model *built on top of* transformer or diffusion architecture assumptions, these architectures clarify the core-ideas so exact param count and structure from model to model differ, e.g.: llama4 architecture, deepseek v3.2 architecture.

- Modular systems naming hell (e.g.: Linux)

- Arch (A gnu/linux distro) : Kernel (Linux/Hurd) + Coreutils (rust, gnu) + Package Manager (pacman) + Init system (systemd) + ...

- Kimi K2 (A gpt-transformer model) : Architecture (Transformer, GPT) + Exact Architecture (elicitation with exact layer count, params, ...) + Optimizer (Muon) + Numerical Precision (FP16/FP32) + Datasets (Pile/Wikipedia/...) + ... 

</v-clicks>

<!--
Presenter Notes for "Commonly Used Jargon":

- Language Model (LM):
  - Vibe check: The OG, could be anything from n-grams to simple neural nets.
  - LM is the foundational concept.

- Large Language Model (LLM):
  - Vibe check: The current rockstars. GPT-4, Claude 3, Gemini, Llama 3.
  Billions (sometimes trillions!) of internal "knobs" the model tunes during training. More knobs = more capacity to learn nuance (and also more to go wrong!).
  Terabytes of text and code. The internet, basically.

  - Emphasize the scale difference (Parameters, Training Data) and its implications.

- Multimodal / VLM:
  - Vibe check: "My LLM can see now! And it's judging my messy desk."
  - This is where things get really interesting – beyond just text. Mention examples.

- SoTA / Cutting-edge / Bleeding-edge:
  - Vibe check: SoTA is what you use for production. Bleeding-edge is what you brag about on X/Twitter.
  - Clarify these terms as they are often used loosely. The "right now" part of SoTA is key due to rapid changes.
-->

---

# A Skewed Map of Language models Realm

<v-clicks>

```mermaid
graph TD
    A[LM Architectures] --> B[Transformer Architecture]
    A --> C[Other Architectures]
    
    B --> D[Encoder-only]
    B --> E[Decoder-only]
    B --> F[Encoder-Decoder]
    
    D --> G[BERT]
    D --> V[Nomic]
    D --> W[GemmaEmbed]
    
    E --> H[<b>GPT by OpenAI</b>]
    E --> I[Llama]
    E --> J[Claude]
    E --> K[Qwen]
    E --> L[Mistral]
    
    F --> N[T5]
    F --> U[Original Transformer]
    
    C --> O[Diffusion]
    C --> P[FNet]
    C --> Q[Mamba]
    C --> R[LSTM]
    C --> S[xLSTM]
    
    classDef highlight fill:#f96,stroke:#333,stroke-width:4px,color:white,font-weight:bold,font-size:60px;
    classDef normal font-size:50px;
    class A,B,E,H,U highlight;
    class C,D,F,G,I,J,K,L,N,O,P,Q,R,S,T,V,W normal;
    linkStyle 0,3,8 stroke:#f96,stroke-width:4px;
```

<AlertBox type="warning">
  This is a skewed map of the LLM realm focused on Assistants and modern LLMs. It's not meant to be exhaustive, but rather to give you a sense of the landscape.

  From now on, "AI" and "LLM" by default points to Decoder-only, Attention-based Transformers, a revision of OpenAI, on Transformers paper called "Attention is all you need" by Google Deepmind
</AlertBox>

</v-clicks>

---
layout: two-cols
---

# So, What is a GPT (Decoding Transformer)?

<!-- TODO! -->

<v-clicks>

- Next word prediction system!
- Tokens!
- Context!
- Parallelism!
- **Relatively** Uncapped Learner!

</v-clicks>

<br>

<Youtube v-click id="9uw3F6rndnA?start=25" width=90% height=40% />

::right::


![alt text](./assets/image-4.png)

---

# I Want to Make One!

Making an LM, is a great project for a resume, but, making an LLM is a financial and technical **mission impossible**. It's a *monumental* undertaking. Not your bachelor's degree project 

<!-- (unless you're a FAANG with a spare million or billion, or of the likes of Terry A Davis). -->

**The Typical Lifecycle:**

<v-clicks>

1. **Data Collection & Curation**
  - Use public datasets on 🤗HuggingFace and Kaggle (won't be enough for SoTA LLMs)
  - Download Everything off the Internet! (including copyrighted material and personal data 😉)
  - Preprocess data (deduplicate, source balancing, interleave, safety filtering, ...)

<br>

2. **Pre-training (Foundation Model)**
  - pick/design architecture, tokenizer, ...
  - Write some code, do some math!
  - Throw your Dataset to your model!
  - Iterate with checkpoints and evaluation, till it learns to predict next words
  - Be in debt for thousands of dollars 🙂

</v-clicks>

---

# I Want to Make One!

<v-clicks>

3. **Fine-tuning (Instruction Tuning & Alignment):**
  - The list of techniques in this section is so long, I won't even try to list them! (we see some later on)

<br>

4.  **Evaluation & Red Teaming:**
  - Test on benchmarks, human evaluations.
  - **Red Teaming**: Try to make it say bad/wrong/biased things to find and fix flaws.

</v-clicks>

<br>

<AlertBox v-click type="success">
This is why building a <b>competitive</b> LLM from scratch is mostly for mega-corps or well-funded research labs. Most of us will be <b>using</b> or <b>fine-tuning existing</b> models. (Including Iranian LLM projects you see in the wild, in which, you can see the puffy, cute <b>tail of llama</b> come off)

But surely, creating LLM is not impossible! Specially with dedicated, smart, dilligent and curious researchers and a few hundred thousands dollars dedicated to R&D (a single pretraining run of a 7B model on limited data, costs roughly <b>5000 dollars</b>!)
</AlertBox>

<!-- 
    - Scrape the internet (Common Crawl), books, code, articles.
    - **Cleaning & Filtering:** Remove PII, hate speech, low-quality content. This is CRUCIAL and hard.
    - *Vibe:* Digital dumpster diving, then gold panning.

        - Goal: Learn general language patterns, grammar, facts, reasoning.
    - Method: Self-supervised learning (e.g., predict masked words, next sentence) on the massive curated dataset.
    - Hardware: Thousands of GPUs running for weeks/months. $$$$$!
    - *Result:* A "foundation model" that knows a lot but isn't very good at following instructions yet (e.g., GPT-3 base).

    Train on high-quality instruction-response pairs (e.g., "Q: Summarize this. A: [summary]"). Makes it follow instructions.
    - **Reinforcement Learning from Human Feedback (RLHF) / Direct Preference Optimization (DPO):**
        - Humans rank different model responses to the same prompt.
        - A "reward model" is trained on these preferences.
        - The LLM is further tuned to maximize scores from the reward model.
        - Goal: Make it more helpful, harmless, and honest.
    - *Vibe:* Sending the raw genius to finishing school.
-->

---

###### borrowed from Julia Turc Youtube channel

![alt text](./assets/image.png)

---

# Ok, Let's Explain Those Tuning Techniques a bit!

<!-- TODO! Finetuning, comments get rid of, fine tune and evaluate clarify, clarfiy what we talk about and what we dont talk about -->

- Fine-tuning Steps!
  - Supervised Fine-Tuning (SFT)
  - Reward Modeling (RLHF / DPO)
  - Preference Tuning (PPO)
  - GPRO
  - CoT
  - Hyper
- Making small models go further
  - Knowledge Distillation
  - Quantization
- Making all models go further
  - Chain of Thought

---

# A Deep-dive for AI Enjoyers

###### An example from Qwen 3 Architecture Blog Post

![](./assets/post-training.png)


<!--
وقتی این مدلا تازه ترین شدن فقط می‌تونند کلمه بعد رو بهمند
-->

---
layout: center
effect: fade
---

# What is in the Magic Box?

## A Lot More, But We Got No Time For:

<br>

- Attention Mechanism
- Cold-Start
- Fine-Tuning In Code
- Math
- Optimizers
- Magic and Spells

## Atleast Today!

---
transition: fade-out
---

# Why LLMs are NOT a Fad

<!-- lossy compression -->

<v-clicks>

### Why Industrial Revolution Succeeded?

It elevated the physical strength of human in specific operations, which boosted manufacturing.

### How Computers Revolution Succeeded?

It elevates the computational strength of human in specific operations, which boosts knowledge work.

### How AI Revolution ... ?

It elevates the cognitive strength of human in specific operations, which automates knowledge work.

<AlertBox type="info">
Just like computers, electricity and steam engines, LLMs are a general solution for a diverse set of problems!
</AlertBox>

</v-clicks>

---
transition: fade-out
layout: center
---

# LLMs are No More Specialist AI, they are Generalist

###### This is my personal opinion

---
transition: fade-out
---

# Why LLMs are NOT a Fad

<v-clicks>

### But, It can't write good code? right?!

If you mean outdated code, sure, it's still an open problem, but in the right, agentic environment, it can write better code than 90% of programmers, and if context-length issue and knowledge recency and confidence issues somehow be resolved or mitigated, it can write better code than 99% of programmers.

### I want to know my code details fully and Engineer it, don't take that away from me!

Well, with that mentality you can never be a good technical-lead engineer, you can't write all the code, you barely can read all the code! if you can't verify LLM code, you can't verify a junior code when you are a senior either! Communicate with LLM, just like a human, assuming that person has dementia and write code super fast! Also with that mentallity, you probably should use ASM for full control! or atleast study compiler throughly

</v-clicks>

---

<div style="display: grid; place-items: center; height: 100%;">

<div>
<h1>An Assumption to Accept</h1>

<div>

LLMs are not mainly a replacement for human intelligence, but they are a replacement for human effort. It doesn't mean they are not intelligent, it means if you are not visionary, you are doomed! You can't out-compete a computer in speed of typing or count of doc pages read per minute, The same way, you can't out-run a car, or out-power a nuclear reactor! Hoping they wont get better by day and eventually catch-up in all aspects, well, sounds quite hopeless to me!

<!-- Unless there be a nuclear war -->

</div>
</div>
</div>

---
layout: center
---

# If Definition of a limit is Being fidgeted with, that limit probably is surpassed already!

<v-clicks>

###### What is Art?


###### Is Human Creativity Special?


###### Is Human Intelligence Special?


###### Can Machine Surpass us?

###### When AI will be, an AGI or ASI?

</v-clicks>

---
layout: center
---

# Singularity is not over the horizon, it's here!

---
layout: center
---

# One more reason, why it's not going anywhere, anytime soon

<img>

<v-clicks>

You are probably using it! and by demand, comes the market!

It's just a matter of lowering the price enough!

So, yeah, this is not NFT bubble (sorry not sorry).

</v-clicks>

---
layout: center
---

# But LLMs, are NOT The Ultimate Silver Bullet!
###### At the very least on their own!

---
layout: center
---

# To Understand Why, Let's Look at Our‌ Differences with LLMs!

---
layout: center
---

# But Before That, Let's Listen to Andrej Karpathy Take on It!

---
transition: fade-out
---

<Youtube id="lXUZvyajciY?start=552" width="100%" height="100%" />

---
transition: fade-out
---

<div style="display: grid; place-items: center; height: 100%;">
  <div>
    <h1>Now, Equipped with Knowledge, Let's Take Another Look to LLM's vs Us: <span v-click>The Dichotomy!</span></h1>
  </div>
</div>

---
layout: two-cols
---

# The Dichotomy!

- **Moravec's paradox, and Biology Architecture**: 
  - Humans and animals, have a lot information baked into their neural pathways
  - For example: A zebra can run a few minutes after birth!
  - There is surely not enough time and data to learn that level of muscle control from ground-up in that time!
  - Approach Difference of AI and Biology <!-- While models, just have a super general algorithm, therefore, more information of humans and animals, are in the architecture itself, humans evolve using meta-architecture of evolution, and this sort of explain-->
  - Evolution meta-architecture
  - Computer Game vs Physics Simulation Example <!-- more on it, example of game development vs physics simulation on computational efficency of realistic things, it takes a few billion years and more than universe GPU to simulate a full universe XD -->

::right::

<div class="pl-4 text-left">

# Which Means

<!-- # Morvac -->

### Humans 💪

- Better motor skills
- General intelligence (More insight on general patterns)
- Agentic and online
- Qualia and Sentience (AI hopefully doesn't)

<br>

### LLM 🦾

- Faster
- Not good in calculations (in compairson to humans and classic computing)
- Less entangled with host machinary
- Simpler architecture

</div>

---
layout: two-cols
---

# The Dichotomy!

- **AI‌ vs Huamns, How we **differ** in learning and memory?**
  - Human's vs LLM's Memory and LLM's duality of learning <!-- Training, vast, slow, costly, deep, vs In-context learning, fast, zero-shot, shallow, ephermal, cheap, e.g.: RAG ||| learning is basically intertwined with active memory , pre-learnt things by evolution, as mentioned before-->
  - Humans superior online learning, sleeping is our online learning superpower? <!-- Abstract Fatal familial insomnia (FFI) is a familial prion disease linked to a mutation of the prion protein gene. Neuropsychological investigations in seven patients with FFI belonging to two different families showed that the main behavioral and neuropsychological features are (1) early impairment of attention and vigilance, (2) memory deficits, mainly of the working memory, (3) impairment of temporal ordering of events, and (4) a progressive dream-like state with neuropsychological and behavioral features of a confusional state. Neuropathologic examination of six patients showed prominent neuronal loss and -->

- **Efficent Online Learning** vs **Efficent In Context Training**
  - Human's Learn Mainly Online

- **Long but Lossy Context ** vs **Accurate but Limited Context Learning/Recall **

::right::

# Which Means

### Humans 💪

- More unified memory sense
- Better and Faster online parameter learner
- No Context Limit (Seamless context management)
- Better reflection 
- More multimodal

<br>

### LLM 🦾

- Super Vast Knowledge
- Super Fast Reader
- Better than average human in-context

- **Context Length Limitations:**
  - Can only "remember" a certain amount of text (the context window). Getting MUCH better (e.g., Gemini 1.5M tokens, Claude 200k), but still a factor for very large tasks. Humans have smart context purge which LLMs currently lack

- **Hallucinations / Confabulation:**
  - They can make stuff up *very confidently*. They are optimized to produce plausible text, not necessarily *true* text

---
layout: two-cols
---

# The Dichotomy!

- **Integrated Actuators and Sensors** vs **Physically Detached by Default**
  - Human are **by now** highly multi-modal compared to MLLM
  - LLM's only IO -aside from its hardware computation foodprint- is Text (or Media for Generative MLLMs)
  - Physical Actuators Make a HUGE Difference! <!--compare a person who needs to make a robot, send its PCB‌ for manufactoring or buy it off amazon to then get napkin while you just standup and take it, assuming you dont have accessibility-->
  - How This Can Become AI's Greatest Strength! <!-- Zero cost copy, infinite limbs -->

---
layout: two-cols
---

# The Dichotomy!

- **Agency, Survival, Mimicing, Thinking Steps**
  - Survive demands agency, mimicing doesn't!
  - The 2 camps: if it quacks like a duck, it is a duck?
  - We are processing, always, LLM's currently (are forced to) play pingponging <!-- mention how its probably not inherent but artifact of lack of data and how it all started + safety>
  - Clear vs Unclear Goal, Machinary and Seperation of Concern
  - A native cron service is up in our brain! <!-- AI can't schedule a thing to do! -->

::right::

# Which Means

### Humans 💪

- Existential and Perpetual Continuation
- More grounded, True world model
‌- Less Hallucination (توهم) and Confabulation (جعل‌ خاطره)
- 

### LLM 🦾

- Cheaper replication of knowledge and scaling
- Easily sweepable for tasks
- 

---

# LLMs are NOT Silver Bullet

They're incredibly powerful, but they're not magic. Think of them as brilliant, hardworking, and sometimes erratic, interns with great general-knowledge but not much of domain-specific experience, and they forget nearly everything the next day you visit them!

<v-clicks>

- **Lack of True Grounding / World Model:**
  - Don't *understand* concepts in a human way. It's sophisticated pattern matching.
  - This leads to subtle (and sometimes not-so-subtle) errors in reasoning or common sense.

- **A Brain Needs Hands (and Eyes, and Ears...):**
  - LLMs generate text. To *do* things in the real world (or digital world), they need to be connected to tools, APIs, databases. (This is where RAG and Agents come in).

</v-clicks>

---
transition: fade-out
---

# But, LLMs are NOT Silver Bullet

<v-clicks>

- **Many Models Does Not Support Multi-Modality Natively**
  - There are tricks used like OCR, for models like deepseek to read pdf, images, ... but understanding all models often comes with a hefty cost on model intelligence, or cost of development, or both!

- **Not Strategic By Default:**
  - They excel at executing well-defined tasks given in the prompt. Complex, multi-step problem-solving requires careful prompting or orchestration (e.g., agentic frameworks).

- **Human Expertise is Still King (or at least, Regent):**
  - **By the time I'm typing this** LLMs are **amplifiers**. They amplify good input and expertise. They also amplify bad input or lack of domain knowledge.
  - Critical thinking, validation, and domain knowledge from humans are ESSENTIAL.

</v-clicks>

<v-clicks>

*Key takeaway:* LLMs are a *component*, not a whole solution. They need to be part of a larger system, often with human oversight.

</v-clicks>

---
transition: fade-out
---

# Model vs. Provider: Who's Who?

This is a common point of confusion, specially for non-technical people. Let's clear it up.

<v-clicks>

- **Model:** The actual AI, the "brain." It's the result of the training process.
    - *Examples:* `GPT-4 Turbo`, `Claude 4 Opus`, `Gemini 2.5 Flash`, `o4-mini` `Llama 3 70B`.
    - These are specific versions with defined capabilities and architectures.

- **Provider (or Product/Service):** The company or platform that hosts the model(s) and provides access, often via an API or a user interface.

</v-clicks>

<AlertBox type="info" title="Analogy Time!">
Think of it like cars:

- **Model:** Ford Focus ST (the specific car with its engine, features)
- **Provider:** The Ford Dealership (where you get it), or Hertz (if you're renting access to drive it).

</AlertBox>

---

# What a Provider Does?

- Run the model(s)
- Provide an API (or a chat interface)
- Tooling for the model
  - Search-engine results for overcoming cut-off
  - Vector DB to query relevant materials
  - Code Runner
- Handle billing and access control
- Handle auxiliary services
  - TTS for hearing answers (another AI!)
  - STT for typing using voice (even another AI!)
  - Memory services
  - Set the right hyperparameters
  - Image gen, ...

---
transition: fade-out
layout: two-cols
---

# Common Hyperparameters (in LLMs)

- Temperature
- Top K
- Top P
- Token/Logit Bias

<div class="half-br"></div>

***

<div class="half-br"></div>

- Max Tokens
- Min Tokens
- Presence Penalty
- Frequency Penalty
- Beam Search Settings

<div class="half-br"></div>

***

<div class="half-br"></div>

- Json Mode and Tool call Settings

::right::

<div class="pl-4 text-left">


Assuming $\text{logit}_i = z_i \quad$ ,
$
\quad \text{scaled logit}_i = \frac{z_i}{T} \quad 
$

We will get:

$$
P_i = \text{softmax}(z/T)_i = \frac{e^{z_i / T}}{\sum_{j=1}^{n} e^{z_j / T}}
$$


Where $n$ is Vocab size and $T$ is Temperature


###### In LLMs Hyperparmeters are often not that intertwined with the model architecture, like some of the other architectures, e.g.: K-means, yet it's interesting to know them


</div>

<!-- formula, card, realization -->

<style>
  .half-br {
    display: block;
    height: 0.5em; /* or 4px, depending on your font size */
    content: "";
  }
</style>

---

<SamplingCard />

---

# Quantization!

- What?
  - **Numerical** and **Lossy** compression on model weights values
- Why?!
  - Even SoTA LLMs Inference is INTENSE! <!-- How much water an llm drink -->
  - Make LM‌ Smaller without training small LMs is cheaper and better!
  - We don't have H100 racks, Remember?
- Why not?!
- How?!
  - Using:
  - Making:
  - Learning:

  

---
layout: center
---

# Quantization Happens in Blocks

---
layout: center
---

# Common Quantizations (GGML Standard)
If you played around with `ollama`, `LM Studio` or similar software These alien-looking phrases after model names, might have confused you greatly, learning them was harder than I expected, so pay close attention!

$qk\_0 \rightarrow$ k bit integer per weight (k is commonly 4, ) with scaling factor (one value for the group)

$qk\_1 \rightarrow$ k bit integer per weight (k is commonly 4, ) with scaling factor (one value for the group) and, a bias value to pinpoint the value more accurately in reconstruction 

<!-- haH? -->

$qk\_k \rightarrow$ 

---

# Let's Get Hands-on

---

# The Art of Asking the Right Questions

To get correct answers, you first need to know, what you want, and how to ask for it.

<script setup>
const promptingTips = [
  {
    title: '0. Language is a medium of information transmission',
    content: 'Therefore a lot of tips on effective human communication work on LLMs as well, maybe even more because they lack the background context of someones persona and desire, as they lack easy personalization method at the moment (e.g.: best we can do for out-of-context learning right now is Reinforcement Fine-Tuning and memories)'
  },
  {
    title: '1. Be Clear and Specific',
    examples: {
      bad: 'I want a website.',
      good: 'I want a portfolio website leveraging cloudflare pages, tailwindcss for styling, my preferred front-end framework is nuxt, paired with astro. here is a list of my projects and documentation of my work...'
    }
  },
  {
    title: '2. Provide Context',
    examples: {
      bad: 'build an mcp-server which provides LLM with capability to read files from my project folder',
      good: 'build an mcp-server which provides LLM with capability to read files from my project folder, here is the mcp-server doc: {full mcp-server doc goes here}'
    }
  },
  {
    title: '3. Keep chat cohesive',
    examples: {
      bad: 'Brainstorm, Write code, ask for life advice and medical advice. all in one chat.',
      good: 'Segregate your tasks into different chats. sharing relevant context between chats only.'
    }
  },
  {
    title: '4. If access to system prompt is available, use it! Specially for things which stay constant across chats',
    examples: {
      bad: '{Keep the defaults and rewrite instructions every, single, time}',
      good: 'You are a helpful and smart programmer and assistant, always use uv with python and bun with javascript when applicable. keep answers simple unless asked for detailed explanation, never shorten the codes with placeholder.',
    }
  },
  {
    title: '5. Iterate and refine your prompt!',
    content: 'Don\'t expect perfect results on the first try. Refine your prompts based on the responses you get. Specially, in agentic systems. In agentic systems, isolate each system prompts, and ensure predictable accuracy and predictable output format.'
  },
  {
    title: '6. Specify Output Format',
    content: 'Tell the model exactly how you want the information is structured and formatted. Use JSON mode if needed.'
  },
  {
    title: '7. Few-shot if you can',
    content: 'Provide examples of the input-output pairs you expect to guide the model, after telling it what to do. These models are next-word prediction machines by default, therefore, this techniques might work even in non-instruction-tuned model'
  },
  {
    title: '8. Tell it what not to do',
    content: 'Sometimes defining boundaries is as important as defining the task.'
  },
  {
    title: '9. Use Delimiters',
    content: 'Separate different parts of your prompt with clear delimiters like ###, """, or ---, some models like claude have explicit guides like using xml for separation of parts'
  },
  {
    title: '10. Use the jargon of the field',
    content: 'Domain-specific terminology helps the model understand the context better, and assumes you are more knowledgeable in the field.'
  },
  {
    title: '11. Use follow-up questions, don\'t ask everything in one chat!',
    content: 'Break complex tasks into a conversation with multiple steps. Either build incrementally and sequentially or in a single chat.'
  },
  {
    title: '12. If things are going too south, pivot into a new chat',
    content: 'Sometimes it\'s better to start fresh than to try to correct a conversation that\'s gone off track.'
  },
  {
    title: '13. Pick the model with right tools and knowledge for your use-case',
    content: 'Different models have different strengths and capabilities.'
  },
  {
    title: '14. Good code practices like module independence',
    content: 'Apply software engineering best practices when asking for code.'
  },
  {
    title: '15. Expect the right thing out of the model',
    content: 'Understand the capabilities and limitations of the model you\'re using.'
  },
  {
    title: '16. Security practices',
    content: 'Be mindful of sensitive information and potential security risks, depending on the provider you are using.'
  },
  {
    title: '17. Don\'t force it to answer first',
    content: 'If model commits to it\'s answer in the start, there is no way back.'
  }
]
</script>

<CardDeck :items="promptingTips" :highlightIndex="0">
</CardDeck>

<!-- Just like humans, AIs cant read your brain (yet) so ask your question -->

---
transition: fade-out
layout: center
---

# Let's Get Hands on for Tooling!

---
transition: fade-out
---

# Random tips for developing with LLM

- learn `git` better than ever!
- LLMs have knowledge cut off! 
- containerize everything!
- Vector DBs

---
transition: fade-out
layout: center
---

# How to Run a Model Locally (Your Private LLM!)

---

# Vibecode!

- Andrej Karpathy
- IDEs
  - Cursor (VSCode Fork)
  - Trae (VSCode Fork)
  - Windsurf (VSCode Fork)
  - Void (VSCode Fork)
- Extensions (for VSCode)
  - Github Copilot
  - RooCode
  - Augment
  - Cline
  - Blackbox
  - Lingma
  - Trae, Windsurf plugins
- Vibe Studios!
  - Bolt.new
  - Lovable
- Github App
  - Code Rabbits

---
transition: fade-out
layout: center
class: text-center
---

# Are you Interested? What's Next? 
Keeping Up & Moving Forward

---
layout: center
---

# Important Tip before moving forward

<br>
<v-clicks>
<AlertBox type="warning">
You <b>cannot</b> learn everything in the world of AI! <br> Aim for <b>strategic sips</b>. after you found what makes you excited. <br> Be <b>deep</b> in something, and follow the news in other
</AlertBox>
</v-clicks>

---

# Opportunities

- Organization of Agents
- Opinionated integrations
- LLM Routing
- MCP Server and Other tooling markets
- n8n-like GUI automations
- Text-defined alternatives to slides, diagrams, ...
- Edge Devices Deployment
- Multi-modality 
- Connecting the dots
- Robotics (Nvidia)
- Chips (Nvidia)
- Fine-tuning for special-domains
- New Architectures?

---
layout: center
---

# How to Expose Yourself to LLM Community

### Follow Key People and Channels:

<br>

<PersonCard :people="aiExperts">
</PersonCard>

<script setup>
const aiExperts = [
  {
    name: "Andrej Karpathy",
    image: "./assets/karpathy.jpg", // Add an image to your project
    description: "Former Director of AI at Tesla, OpenAI founding member. Known for excellent educational content on neural networks and deep learning.",
    platforms: [
      { type: "youtube", handle: "@AndrejKarpathy - 'Neural Networks: Zero to Hero' series" },
      { type: "twitter", handle: "@karpathy" },
      { type: "github", handle: "karpathy" }
    ]
  },
  {
    name: "Dwarkesh Patel",
    image: "./assets/dwarkesh.png",
    description: "Host of the Dwarkesh Podcast (formerly The Lunar Society). Conducts in-depth technical interviews with leading AI researchers, entrepreneurs, and thought leaders including John Carmack, Dario Amodei, Andrej Karpathy and Mark Zuckerberg.",
    platforms: [
      { type: "youtube", handle: "@DwarkeshPatel - Deep-dive AI research interviews" },
      { type: "twitter", handle: "@dwarkesh_sp" },
      { type: "website", handle: "dwarkeshpatel.com - Podcast episodes and transcripts" }
    ]
  },
  {
    name: "Yann LeCun",
    image: "./assets/yanlecun.jpeg", 
    description: "Chief AI Scientist at Meta, Turing Award winner. Pioneer in deep learning and computer vision.",
    platforms: [
      { type: "twitter", handle: "@ylecun" },
      { type: "linkedin", handle: "yann-lecun" }
    ]
  },
  {
    name: "Yannic Kilcher",
    image: "./assets/yannic_kilcher.jpg",
    description: "ML researcher and YouTube educator known for in-depth paper breakdowns. Former PhD from ETH Zurich, CTO of DeepJudge. Explains cutting-edge research with clarity and occasional humor.",
    platforms: [
      { type: "youtube", handle: "@YannicKilcher - Paper explanations and ML News" },
      { type: "twitter", handle: "@ykilcher" },
      { type: "website", handle: "ykilcher.com" }
    ]
  },
  {
    name: "AI Explained (Philip)",
    image: "./assets/philip_ai_explained.jpg",
    description: "Creator of AI Explained YouTube channel and runs AI Insiders community. Breaks down latest AI research papers, models, and benchmarks with detailed analysis.",
    platforms: [
      { type: "youtube", handle: "@aiexplained-official - Model breakdowns and analysis" },
      { type: "newsletter", handle: "Signal to Noise newsletter" },
      { type: "community", handle: "AI Insiders - 1000+ professionals" }
    ]
  },
  {
    name: "Lex Fridman",
    image: "./assets/lex_fridman.jpg",
    description: "Host of one of the most popular AI/tech podcasts with 4+ hour deep conversations. MIT researcher. Interviews range from AI researchers to tech leaders and philosophers.",
    platforms: [
      { type: "youtube", handle: "@lexfridman - Long-form technical interviews" },
      { type: "twitter", handle: "@lexfridman" },
      { type: "spotify", handle: "Lex Fridman Podcast" }
    ]
  },
  {
    name: "Jim Fan",
    image: "./assets/jimfan.jpg",
    description: "AI researcher at NVIDIA. Known for insights on multimodal AI and cross point of robotics and AI, has some great presentations specially on seqouia capital",
    platforms: [
      { type: "twitter", handle: "@DrJimFan" }
    ]
  },
  {
    name: "Ilya Sutskever",
    image: "./assets/ilya_sutskever.jpg",
    description: "Co-founder and Chief Scientist of OpenAI (former), now founding Safe Superintelligence Inc. Key architect behind GPT models. Deep expertise in AI safety and AGI development.",
    platforms: [
      { type: "twitter", handle: "@ilyasut" },
      { type: "company", handle: "Safe Superintelligence Inc. - New AI safety venture" },
      { type: "appearances", handle: "Frequent podcast guest on AI future" }
    ]
  },
  {
    name: "AI Research Organizations",
    description: "Follow official blogs and publications from leading AI labs.",
    platforms: [
      { type: "blog", handle: "OpenAI, Anthropic, Google DeepMind, Meta AI, Hugging Face" }
    ]
  },
  {
    name: "AI Snake Oil",
    description: "Critical perspectives on AI hype and limitations by Arvind Narayanan & Sayash Kapoor.",
    platforms: [
      { type: "website", handle: "aisnakeoil.com" }
    ]
  },
  {
    name: "arXiv Papers",
    description: "Look for highly cited/trending papers in Computation and Language (cs.CL), AI (cs.AI), and Machine Learning (cs.LG).",
    platforms: [
      { type: "arxiv", handle: "cs.CL, cs.AI, cs.LG categories" }
    ]
  },
  {
    name: "AI Newsletters",
    description: "Regular curated updates on AI developments.",
    platforms: [
      { type: "website", handle: "AI Alignment Newsletter, The Batch, Last Week in AI" }
    ]
  },
  {
    name: "AI Communities",
    description: "Join discussions with other AI enthusiasts and professionals.",
    platforms: [
      { type: "website", handle: "Hugging Face forums, Reddit (r/LocalLLaMA, r/MachineLearning), Discord servers" }
    ]
  },
  {
    name: "Hands-on Experience",
    description: "The best way to learn is by doing. Pick a model, try to build something small.",
    platforms: [
      { type: "github", handle: "Try open-source models and tools" }
    ]
  }
]
</script>

---

**How to "Learn" More & Stay Updated:**
<v-clicks>

- **Read Blog Posts:** OpenAI, Anthropic, Google AI blogs. AI Snake Oil by Arvind Narayanan & Sayash Kapoor for critical perspectives.
- **Key Papers on arXiv:** Look for highly cited / trending papers in `cs.CL` (Computation and Language), `cs.AI`, `cs.LG` (Machine Learning).
- **Newsletters:** AI Alignment Newsletter, The Batch, Last Week in AI.
- **Communities:** Hugging Face forums, Reddit (r/LocalLLaMA, r/MachineLearning), Discord servers.
- **Experiment!** The best way to learn is by doing. Pick a model, try to build something small.

</v-clicks>

**Future Vibe (Speculative but Grounded):**
<v-clicks>


- **Deeper Multimodality:** LLMs that seamlessly understand and generate across text, image, audio, video, maybe even other sensor data.
- **Longer, More Effective Context:** Less "goldfish memory."
- **Improved Reasoning & Reliability:** Fewer hallucinations, better step-by-step logic.
- **More Capable Agents:** LLMs that can reliably use tools, plan, and execute complex tasks with less human hand-holding.
- **Personalization & Specialization:** Models fine-tuned for specific domains, industries, or even individuals.

</v-clicks>

<!-- با همه فاجعه شدن ها هنوز خیلی خوبه-->

---


---


<div style="display: grid; place-items: center; height: 100%;">
  <div>
      <h1>To Predict the Future, <span v-click> We Shall Look Deeper to Past and Present!</span><br><br><span v-click>Let's Examine Current Trends in Papers and Models</span></h1>
  </div>
</div>

---
transition: fade-out
---

# New LLM and Tangental AI Trends

<v-clicks>

- Vocal native models
  - Voxtral <span class="text-gray-400">(by Mistral)</span>
- Robotic models
  - MolmoAct <span class="text-gray-400">(by Ai2)</span>
  - OmniVinci <span class="text-gray-400">(by Nvidia)</span>
- Hybrid-Reasoning Models (HRMs)
  - DeepSeek v3.1 <span class="text-gray-400">(by DeepSeekAI, Explicit Token-Based Switching)</span>
  - DeepSeek v3.2 Exp <span class="text-gray-400">(by DeepSeekAI, Explicit Token-Based Switching)</span>
  <!-- Qwen,  -->
  - GLM 4.5 and 4.6 
  - SmolLM3
  - Qwen 3
- Prompt Compression and Extension Techniques
- Agentic Models

</v-clicks>

---

# Homework :)

<v-clicks>

1.  **Try a local model:** Install ollama, LM Studio, open-webui or all of the above! Download Llama 3 8B or Phi-3-mini. Chat with it.
2.  **Experiment with Prompt Engineering:** Take a task you do regularly (writing an email, summarizing text, generating code snippets) and try to get an LLM (local or cloud) to do it via careful prompting. Iterate at least 5 times on your prompt.
3.  **Explore one LLM-powered tool:** If you code, try Cursor or GitHub Copilot for a week. If you research, try Perplexity.
4.  **Read one "intro to RAG" article or watch a short video.** Understand the basic concept.

</v-clicks>


---
layout: center
class: text-center
---

# Questions? & Thank You!

Let's vibe and discuss.

<br>

This is just a dip into the ocean of LLM literacy, a lot left unsaid, and a lot remains in fog of mystery. Stay curious, keep experimenting, and don't be afraid to ask a **dumb** questions. The field is new for everyone!

Your presence mean to me as much as count of GPT4.5 parameters

<div class="mt-8">
Connect with me:

<br>
<br>

<img src="./assets/qr-code.svg" class="mx-auto" style="width: 200px; height: 200px;" alt="QR Code">

</div>



