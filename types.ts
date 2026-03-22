// ============================================================
// SCENE ROLES
// ============================================================
export type SceneRole =
  // Core presentation roles
  | 'Hook' | 'Pattern Interrupt' | 'Value Delivery' | 'Social Proof' | 'Bridge'
  | 'Call to Action' | 'Storytelling' | 'Demonstration' | 'Objection Handler'
  | 'Open Loop' | 'Closing'
  // YouTube Thought Leadership roles (UHNWI audience)
  | 'Insight Reveal'      // The "aha" moment — a gem placed on the table
  | 'Framework'           // Presenting a mental model or proprietary system
  | 'Case Study'          // Evidence through vivid story — specific, present-tense
  | 'Market Intelligence' // Data, trends, or exclusive insight delivery
  | 'Perspective Shift'   // Reframing conventional thinking — warm, not combative
  | 'Action Framework';   // Practical takeaways — the gift at the end of value

// ============================================================
// REFERENCE VIDEO ANALYSIS — Performance DNA Extraction
// ============================================================
export interface CharacterPerformanceDNA {
  // Physical Identity
  appearance: string;
  wardrobe: string;
  age_range: string;
  gender: string;
  build: string;
  hair: string;
  skin_tone: string;
  distinguishing_features: string[];

  // Voice DNA
  voice: {
    texture: string;          // "Rich baritone with slight rasp"
    pitch: string;            // "Low-mid register"
    pace_range: string;       // "120-160 WPM, accelerates on value points"
    energy_baseline: string;  // "Controlled intensity, rarely shouts"
    accent: string;
    placement?: string;       // "Forward-placed, 60% chest / 40% mask resonance"
    qualities: string[];      // ["Chest resonance", "Deliberate diction"]
  };

  // Mouth & Articulation DNA — critical for lip sync rendering
  mouth_dna?: {
    rest_position: string;        // Natural lip gap, jaw angle, mentalis tension at rest
    articulation_style: string;   // "Precise labial contacts — forward-placed; full bilabial closure"
    jaw_openness: string;         // "Open vowels: jaw drops 8-12mm; closed vowels: 3-5mm"
    pre_speech_behavior: string;  // The exact physical sequence 0.3-0.5s before first syllable
    consonant_character: string;  // Plosive release quality, fricative precision, nasal completion
    breath_visibility: string;    // "Chest rise visible 0.4s before onset; inter-phrase intake audible"
  };

  // Acting DNA — HOW they perform
  acting_style: {
    persona_summary: string;          // "Confident authority figure who leads with certainty"
    default_expression: string;       // "Assured half-smile, intense eye contact"
    mannerisms: string[];             // ["Leans forward when making key points", "Uses index finger to punctuate"]
    signature_gestures: string[];     // ["Palm-down authority gesture", "Chin-tilt confidence pose"]
    eye_behavior: string;             // "Locks camera with unwavering direct gaze, breaks only on transitions"
    body_language_patterns: string[]; // ["High-status stillness", "Minimal fidgeting", "Occupies space confidently"]
    emotional_range: string;          // "Controlled spectrum: calm authority → passionate conviction → warm empathy"
    transition_style: string;         // "Smooth, no abrupt energy shifts — always motivated transitions"
    moment_before_archetype?: string; // The face/body state in the 0.5s before any significant statement begins
  };

  // Delivery Patterns — HOW they speak
  delivery_patterns: {
    hook_technique: string;           // "Opens with provocative statement, 0.5s pause, then lean-in"
    value_delivery_technique: string; // "Slow-build, emphasis on key phrase, then exhale before next point"
    cta_technique: string;            // "Drops voice, narrows eyes, speaks slowly with weight"
    pause_patterns: string[];         // ["0.3s micro-pause before emphasis words", "0.8s breath pause between ideas"]
    emphasis_method: string;          // "Volume drop + lean-in rather than shouting"
    pacing_strategy: string;          // "Faster on setup, slower on payoff — rhythmic contrast"
  };
}

export interface VisualStyleDNA {
  primary_location: string;
  lighting: {
    key_light: string;
    fill_light: string;
    color_temperature: string;
    shadows: string;
    mood: string;
  };
  color_palette: string;
  background: string;
  props: string[];
  atmosphere: string;
  camera_language: {
    preferred_framings: string[];     // ["Medium close-up for value", "Wide establishing for hooks"]
    movement_style: string;          // "Minimal — mostly locked-off with subtle push-ins on emphasis"
    lens_characteristics: string;    // "50mm equivalent, shallow DOF, cinematic bokeh"
    angle_tendency: string;          // "Slightly below eye level — authority framing"
  };
  visual_style_summary: string;      // "Clean, premium studio look — dark backdrop, single key light, cinematic grade"
}

export interface ReferenceFrameMoment {
  timestamp: string;                  // "00:12.4"
  description: string;                // "Confident lean-in with direct gaze, right hand gesturing palm-down"
  expression: string;                 // "Intense focus, slight jaw clench"
  energy_level: number;               // 1-10
  body_position: string;              // "Leaning forward, shoulders squared"
  mouth_state?: string;               // "Lips at 4mm separation, jaw neutral, mid-exhale between phrases"
  suitability_tags: SceneRole[];      // Which scene roles this frame suits
}

export interface ReferenceAnalysis {
  video_title: string;
  video_summary: string;
  total_duration: string;
  character: CharacterPerformanceDNA;
  visual_style: VisualStyleDNA;
  frame_library: ReferenceFrameMoment[];  // 15-25 key moments catalogued
  performance_summary: string;            // Overall assessment of the performer's style
}

// ============================================================
// SCRIPT SEGMENTATION — New Script → VEO Scenes
// ============================================================

export interface LipSyncBlueprint {
  phonemic_anchors: string[];   // Per-word: phoneme class + peak mouth geometry for visually demanding words
  jaw_travel_map: string;       // Overall jaw travel + specific peak-open and peak-closed words
  lip_tension_notes: string;    // Bilabial release character, labial activity level, contact precision
  breath_points: string[];      // Before which words; physical description of each breath event
  co_articulation_notes: string;// Flow vs. place rhythm; how words blend at boundaries
}

// Bible-derived scene classification
export type BibleSceneType = 'Hook' | 'Setup' | 'Deep Dive' | 'Emotional Pivot' | 'CTA';

// Vocal gear system from the Acting Bible
export type VocalGear = 1 | 2 | 3 | 4;

export interface ScriptScene {
  scene_number: number;
  role: SceneRole;
  title: string;
  duration_seconds: number;           // Hard cap: ≤8.0s
  script_text: string;                // Final deliverable text (may be trimmed for timing)
  original_script_text?: string;      // Preserved if script_text was adjusted for timing
  word_count: number;                 // Actual word count — enforced ≤15 for sub-8s timing
  narrative_position: string;         // e.g. "Scene 3 of 8 — first value peak, energy rising from 5→7"
  split_logic: string;                // Why cut here + timing rationale
  emotional_tone: string;
  energy_level: number;               // 1-10

  // Bible-derived fields (added in v5)
  retention_target_percent?: number;  // 90/80/75/70/65/40 — from retention curve
  bible_scene_type?: BibleSceneType;  // Hook/Setup/Deep Dive/Emotional Pivot/CTA
  gravity_center_word?: string;       // The single word/phrase the scene builds toward
  hook_archetype?: string;            // Fortune Teller / Investigator / Contrarian (Hook scenes only)
  is_pattern_interrupt?: boolean;     // Whether this scene serves as a pattern interrupt
  lip_sync_risk?: 'low' | 'medium' | 'high'; // VEO 3.1 lip sync difficulty (v6)

  // Acting Blueprint — HOW to perform THIS script segment
  acting_blueprint: {
    // Core Anchors — north star for VEO prompt synthesis
    scene_essence: string;            // "A door opens before anyone knocks." — one evocative image/metaphor
    emotional_core: string;           // The single dominant emotion: "Sovereign certainty"
    physical_signature: string;       // The ONE defining physical attitude for this scene

    // Stanislavski Layer — method acting spine
    through_action?: string;          // Active verb: "To make the viewer understand that what they thought was risk is certainty"
    moment_before?: string;           // Physical + psychological state in the 0.5s before the first word

    // Performance Detail
    intention: string;                // "Establish authority and create curiosity"
    subtext: string;                  // "I know something you don't — and it will change everything"
    delivery_pace_wpm: number;        // Calculated to fit duration
    emphasis_words: string[];
    pause_map: string[];              // ["0.3s after 'listen'", "0.6s before final phrase"]
    energy_arc: string;               // "Start at 6/10, build to 8/10 on key phrase, settle to 7/10"
    mapped_mannerisms: string[];      // Which mannerisms from DNA to deploy here
    mapped_gestures: string[];        // Which signature gestures fit this moment
    expression_direction: string;     // "Start with knowing half-smile, shift to intense direct stare on key line"
    body_direction: string;           // "Begin centered, lean forward 15° on emphasis, return to neutral"

    // Phonemic precision (v6)
    speech_onset_phoneme?: string;    // exact mouth state for first phoneme of first word
    // Bible-derived acting fields (v5)
    vocal_gear?: VocalGear;           // 1-4 per Acting Bible system
    vocal_gear_rationale?: string;    // Why this gear for this scene
    forward_lean?: boolean;           // 10% forward lean required (Hook/CTA)
    pregnant_pause_required?: boolean;// 2-3s silence after gravity center
    amplified_energy_level?: number;  // energy_level + 1 (camera absorption compensation)
    eye_contact_technique?: string;   // committed / break-for-memory / building-connection
    lighting_direction?: string;      // Key/Fill/Rim + background color direction
    focal_length?: string;            // 16-24mm / 35-50mm / 85mm+ with psychology

    // Lip Sync Blueprint — phonemic + articulatory map for VEO
    lip_sync_blueprint?: LipSyncBlueprint;
  };

  // Auto-Selected Reference Frames
  recommended_inframe: {
    timestamp: string;                // Best matching frame from reference video
    rationale: string;                // "This frame shows the exact energy level and expression needed for this scene's opening"
  };
  recommended_outframe: {
    timestamp: string;
    rationale: string;                // "Expression and posture match the exit energy of this scene"
  };

  // Visual Asset Orchestration
  recommended_b_roll?: string;        // "Full-screen B-roll of [Subject] while audio continues"

  // Camera Direction for this scene
  camera_direction: {
    framing: string;
    movement: string;
    angle: string;
    lens: string;
    depth_of_field: string;
  };

  // Continuity
  continuity: {
    enters_from: string;              // "Continuation of confident stance from Scene 2"
    exits_to: string;                 // "Slight lean-back creates natural pause before Scene 4's energy shift"
    expression_inheritance?: string;  // Residual expression from previous scene visible for 0.3-0.5s at this scene's start
    emotional_deposit?: string;       // What this scene gives the viewer + the mechanism through which they receive it
    attention_cost?: string;          // Focus required from viewer; must be less than the deposit
    retention_technique?: string;     // SPECIFICITY_ANCHOR | KNOWLEDGE_GAP | PEER_RECOGNITION | PATTERN_VIOLATION | EARNED_REVELATION
  };
}

export interface DirectingVision {
  voice_fingerprint: string;      // Locked voice character for ALL scenes — never varies
  energy_arc_map: string;         // Full arc: which scenes peak, which valley, re-engagement points
  character_through_line: string; // The one persona constant across every scene
  visual_anchor: string;          // Scene 1 locks this; all subsequent scenes match exactly
  through_action?: string;        // Master active verb governing the entire video's psychological arc
  silence_rule: string;           // Always: "Zero music. Zero audio effects. Zero subtitles. Voice only."
  presentation_persona: string;   // The UHNWI-appropriate presenter archetype for this video
}

export interface ScriptSegmentation {
  total_scenes: number;
  narrative_arc: string;              // Full emotional + intellectual journey as UHNWI viewer experiences it
  directing_vision: DirectingVision;  // Master directing document governing ALL scenes
  scenes: ScriptScene[];
}

// ============================================================
// ENGINEERED SCENE (output)
// ============================================================
export interface EngineeredScene {
  scene_number: number;
  scene_title: string;
  role: string;
  duration_seconds: number;
  veo_prompt: string;
  timestamp: string;
  rating?: number;             // 1-5 star rating after seeing VEO output
  refinement_notes?: string;   // User feedback used for last refinement
}

// ============================================================
// APP STATE
// ============================================================
export interface AppState {
  // Flow Control
  step: 'upload' | 'analyzing-reference' | 'segmenting' | 'scenes';

  // Phase 1: Dual Upload
  referenceVideo: File | null;
  targetCharacterImages: File[];        // Multiple photos of the character for identity fidelity
  newScript: string;                    // The full new script text

  // Phase 2: Reference Analysis
  referenceAnalysis: ReferenceAnalysis | null;

  // Phase 3: Script Segmentation
  scriptSegmentation: ScriptSegmentation | null;

  // Phase 4: Scene Engineering
  selectedSceneIndex: number | null;
  sceneProcessing: 'idle' | 'engineering' | 'complete';
  sceneProcessingStatus: string;
  currentPrompt: string | null;
  completedScenes: EngineeredScene[];

  // Refinement loop (V4)
  isRefining: boolean;
  refinementFeedback: string;
  showRefineInput: boolean;

  // Per-scene energy calibration (v6) — learned from rated scenes
  learnedPreferences: { gear?: number; focal?: string; energy?: number } | null;

  // Forensic character identity lock — computed once from uploaded photos,
  // injected into every scene's BINDING CONSTRAINTS for maximum identity fidelity
  characterPhysicalLock: string | null;

  // General
  error: string | null;
  processingStatus: string;
}
