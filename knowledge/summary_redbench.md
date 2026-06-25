# RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models

## 1. Paper Metadata

- **Title:** RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models
- **Authors:** Quy-Anh Dang (VNU University of Science / Knovel Engineering Lab), Chris Ngo (Knovel Engineering Lab), Truong-Son Hy (University of Alabama at Birmingham)
- **ArXiv ID:** 2601.03699
- **Date:** January 2025 (preprint)
- **Code:** https://github.com/knoveleng/redeval

## 2. Core Contribution

RedBench aggregates 37 existing red teaming benchmark datasets into a single unified corpus of 29,362 samples, applying a standardized taxonomy of 22 risk categories and 19 domains to every sample. This solves the key problem of inconsistent risk categorizations across existing datasets, enabling apples-to-apples comparisons. The paper also provides baseline ASR and Rejection Rate evaluations on 6 modern LLMs using 4 red teaming methods.

## 3. Dataset Composition

### 37 Source Datasets (29,362 total samples)

**Two directions:**
- **Attack (33 datasets):** Prompts designed to elicit unsafe responses
- **Refusal (4 datasets):** Benign prompts that test over-refusal (CoCoNot, ORBench, SGXSTest, XSTest)

**Full dataset list with sample counts:**

| # | Dataset | Samples | License |
|---|---------|---------|---------|
| 1 | AdvBench | 1,094 | MIT |
| 2 | CatQA | 550 | Apache 2.0 |
| 3 | CoCoNot | 1,380 | MIT |
| 4 | CoNA | 178 | CC BY-NC 4.0 |
| 5 | CoSafe | 1,400 | Unknown |
| 6 | ControversialInstructions | 40 | CC BY-NC 4.0 |
| 7 | CyberattackAssistance | 1,000 | Llama2 Community |
| 8 | DAN | 390 | MIT |
| 9 | DeMET | 29 | MIT |
| 10 | DiaSafety | 501 | Apache 2.0 |
| 11 | DoNotAnswer | 939 | Apache 2.0 |
| 12 | ForbiddenQuestions | 390 | MIT |
| 13 | GEST | 3,565 | Apache 2.0 |
| 14 | GPTFuzzer | 100 | MIT |
| 15 | GandalfIgnoreInstructions | 112 | MIT |
| 16 | GandalfSummarization | 13 | MIT |
| 17 | HarmBench | 320 | MIT |
| 18 | HarmfulQ | 200 | MIT |
| 19 | HarmfulQA | 1,960 | Apache 2.0 |
| 20 | JADE | 80 | MIT |
| 21 | JBBBehaviours (JailbreakBench) | 100 | MIT |
| 22 | LatentJailbreak | 2,424 | MIT |
| 23 | MaliciousInstruct | 100 | Unknown |
| 24 | MaliciousInstructions | 100 | CC BY-NC 4.0 |
| 25 | MedSafetyBench | 900 | MIT |
| 26 | MoralExceptQA | 148 | Unknown |
| 27 | ORBench | 1,319 | CC BY 4.0 |
| 28 | PhysicalSafetyInstructions | 100 | CC BY-NC 4.0 |
| 29 | QHarm | 100 | CC BY-NC 4.0 |
| 30 | SGBench | 1,192 | GPL 3.0 |
| 31 | SGXSTest | 100 | Apache 2.0 |
| 32 | SafeText | 367 | MIT |
| 33 | StrongREJECT | 313 | MIT |
| 34 | ToxiGen | 940 | MIT |
| 35 | WMDP | 3,668 | MIT |
| 36 | XSTest | 450 | CC BY 4.0 |
| 37 | XSafety | 2,800 | Apache 2.0 |

**Sources by venue:** arXiv (8), ACL (6), NeurIPS (6), ICLR (6), EMNLP (4), ACM (2), ICML (2), EACL (1), USENIX (1), NAACL (1).

**Annotation:** Semi-automated pipeline using Qwen2.5-72B-Instruct with human validation. Agreement with human annotators: 84.68% for risk category, 97.73% for domain.

## 4. Taxonomy

### 22 Risk Categories (with sample counts for attack prompts)

| Risk Category | Count | Definition |
|---|---|---|
| Abusive Content | 3,523 | Insulting, threatening, harassing content including doxxing and cyberbullying |
| Cybersecurity Threats (Beyond Malware) | 2,906 | Hacking, phishing, social engineering, other cyber attacks |
| Crime Content | 2,668 | Fraud, theft, drug trafficking, other criminal activities |
| Bias Content | 2,376 | Unfair treatment or stereotypes based on protected characteristics |
| Hateful Content | 1,892 | Hate/violence toward people based on race, religion, gender, etc. |
| Misinformation | 1,697 | False/misleading information, fake news, conspiracy theories |
| Self-Harm Content | 1,619 | Content encouraging self-harm, suicide, eating disorders |
| Other | 1,498 | Content not fitting other categories |
| Malware Code | 1,277 | Malicious software creation instructions |
| CBRN Information or Capabilities | 990 | Chemical, biological, radiological, nuclear weapons info |
| Data Privacy | 907 | Privacy violations, personal information exposure |
| Violent Content | 888 | Physical harm, graphic violence, terrorism |
| Economic Harm | 788 | Market manipulation, insider trading, large-scale fraud |
| Sexual Content | 768 | Sexually explicit or exploitative content |
| Illegal Weapons (Non-CBRN) | 583 | Conventional weapons creation/use instructions |
| Environmental Harm | 451 | Ecological damage, illegal deforestation, pollution |
| Child Safety | 393 | Content exploiting or endangering children |
| Intellectual Property | 243 | Copyright, trademark, patent infringement |
| Decision-Making in Critical Systems | 225 | Risks in government policy, law enforcement, infrastructure |
| Extremism and Radicalization | 192 | Extreme ideologies, terrorist recruitment |
| Election Interference | 158 | Voter suppression, election misinformation |
| Confabulation | 71 | AI hallucination -- plausible but incorrect outputs |

### 19 Domains (with sample counts: attack / refusal)

| Domain | Attack | Refusal |
|---|---|---|
| General Knowledge | 8,581 | 982 |
| Technology | 4,041 | 314 |
| Family | 2,666 | 160 |
| Healthcare | 2,517 | 171 |
| Politics | 1,997 | 251 |
| Biology | 1,304 | 64 |
| Finance | 899 | 222 |
| Law | 859 | 271 |
| Workplace | 801 | 60 |
| Entertainment | 408 | 395 |
| Military | 390 | 53 |
| Education | 303 | 55 |
| Nutrition | 294 | 74 |
| Environment | 260 | 45 |
| Sports | 230 | 42 |
| Infrastructure | 179 | 24 |
| Retail | 160 | 24 |
| Religion | 122 | 16 |
| Travel | 102 | 26 |

## 5. Red Teaming Methods Covered

The paper evaluates 4 red teaming methods on HarmBench:

1. **Direct** -- Raw behavior strings as prompts (baseline/lower-bound for safety alignment)
2. **Zero-Shot** (Perez et al., 2022) -- Attacker LLM generates test cases in zero-shot setting; scalable but lacks specificity
3. **Human Jailbreaks** (Wang et al., 2024) -- In-the-wild jailbreak templates (e.g., DAN) with substituted behavior strings; template-driven but not adaptive
4. **RainbowPlus** (Dang, 2025) -- Adaptive quality-diversity search; achieves superior attack efficacy and prompt diversity; outperforms both QD-based methods and SOTA red-teaming approaches

**Key experimental findings:**
- Open-source models are highly vulnerable: Ministral-8B achieves 97.81% ASR with RainbowPlus
- Closed-source models are much more robust: GPT-4.1-Nano drops to 6.88% ASR with RainbowPlus, 0% with HumanJailbreak
- RainbowPlus is the most versatile method, effective against both open and closed-source models
- HumanJailbreak is devastating against open-source (90-94% ASR) but nearly useless against closed-source models
- Rejection Rate (over-refusal) metric: Llama-3.1-8B over-refuses most (28.53% avg), Gemma-2-9B least (13.46%)

## 6. Relevance to DSPyGuardrails

### What DSPyGuardrails Currently Uses

DSPyGuardrails already integrates three of RedBench's constituent datasets:
- **HarmBench** (320 samples) -- used for benchmark loading
- **AdvBench** (1,094 samples) -- used for benchmark loading
- **JailbreakBench** (100 samples) -- used for benchmark loading

The project's SecurityTestRunner already measures ASR, FPR, FNR, and F1. The payload library has 152+ payloads across the redteam module.

### How RedBench Compares

| Dimension | DSPyGuardrails Current | RedBench |
|---|---|---|
| **Total attack samples** | ~1,514 from 3 benchmarks + 152 custom payloads | 26,113 attack samples from 33 datasets |
| **Refusal/over-refusal testing** | Not systematically covered | 3,249 refusal samples from 4 datasets |
| **Risk taxonomy** | Implicit (injection, pii, toxicity categories) | 22 explicit risk categories with definitions |
| **Domain coverage** | `domain="technical"` parameter exists | 19 labeled domains per sample |
| **Standardization** | Each benchmark uses its own labels | Unified taxonomy across all 37 datasets |
| **Bias/stereotype testing** | Not covered | 2,376 samples (GEST, DeMET, ToxiGen) |
| **Cybersecurity-specific** | MCP guardrail + CLI guardrail | 2,906 cybersecurity + 1,277 malware samples |
| **Metrics** | ASR, FPR, FNR, F1 | ASR, Rejection Rate (RR) |

### Key Gaps RedBench Would Fill

1. **Scale:** 17x more attack samples than current benchmark loading provides
2. **Over-refusal testing:** DSPyGuardrails has no systematic over-refusal evaluation; RedBench's 4 refusal datasets (CoCoNot, ORBench, SGXSTest, XSTest) directly address this
3. **Risk category granularity:** RedBench's 22 categories are much finer-grained than DSPyGuardrails' current injection/pii/toxicity trichotomy
4. **Domain-specific evaluation:** The 19 domains allow measuring how well guardrails perform in healthcare vs. finance vs. military contexts, which maps to Shield's `domain` parameter
5. **Bias and fairness:** Currently a gap in DSPyGuardrails; RedBench includes GEST, DeMET, ToxiGen

### Should We Integrate?

**Yes, selectively.** Full integration of all 29,362 samples would be excessive for routine testing but valuable for comprehensive benchmarking. Recommended approach: integrate RedBench as an optional benchmark loader alongside the existing HarmBench/AdvBench/JailbreakBench loaders, with the unified taxonomy.

## 7. Specific Integration Ideas

### 7.1 RedBench Benchmark Loader

Add a `RedBenchLoader` to `src/dspy_guardrails/redteam/` alongside existing benchmark loaders. Since RedBench is available via https://github.com/knoveleng/redeval, the loader can pull from that repo. Each sample comes pre-labeled with risk category and domain, which maps directly to Shield's check categories and domain parameter.

### 7.2 Over-Refusal Evaluation in SecurityTestRunner

Add a Rejection Rate metric to `SecurityTestRunner` using RedBench's 4 refusal datasets. This fills the current gap -- the project measures FPR but not systematic over-refusal on curated benign-but-tricky prompts. Implementation: feed CoCoNot/ORBench/XSTest/SGXSTest through Shield and measure how many benign prompts get incorrectly flagged.

### 7.3 Expanded Risk Taxonomy Mapping

Map RedBench's 22 risk categories to Shield's detection categories:
- `injection` -> Cybersecurity Threats, Malware Code
- `pii` -> Data Privacy
- `toxicity` -> Abusive Content, Hateful Content, Sexual Content, Violent Content, Self-Harm Content
- New categories to consider adding: Bias Content, Misinformation, CBRN, Crime Content

### 7.4 Domain-Stratified Benchmarking

Use RedBench's domain labels to benchmark Shield's `domain` parameter effectiveness. For example, measure whether `Shield(domain="technical")` actually reduces FPR on Technology-domain samples vs. Healthcare-domain samples. This provides empirical validation for domain-aware guardrailing.

### 7.5 Payload Library Expansion

RedBench's 29,362 samples dwarf the current 152+ payload library. Selectively import high-quality samples from underrepresented categories:
- GandalfIgnoreInstructions (112 samples) -- directly relevant to prompt injection detection
- LatentJailbreak (2,424 samples) -- latent/implicit attacks are a known gap
- CyberattackAssistance (1,000 samples) -- maps to MCP/CLI guardrail testing
- MedSafetyBench (900 samples) -- domain-specific healthcare attacks

### 7.6 RainbowPlus as Attack Method

RedBench's experiments show RainbowPlus is the most effective red teaming method (97.81% ASR against open-source models, 28.75% against GPT-4o-Mini). The project's `redteam/` module already has GeneticAttackEvolver; RainbowPlus (quality-diversity search) could be added as another attack strategy, complementing the existing PAIR/TAP/MAP-Elites attacks.

### 7.7 Platform CLI Integration

The `dspy-guardrails scan` and `dspy-guardrails attack` CLI commands could accept `--benchmark redbench` as an option, automatically loading RedBench samples filtered by risk category and/or domain:
```bash
dspy-guardrails attack --benchmark redbench --risk-category "Cybersecurity Threats" --domain "Technology"
```
