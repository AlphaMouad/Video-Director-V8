import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  analyzeReferenceVideo,
  segmentScript,
  refineScenePrompt,
  runVeoPromptAgent,
  buildScene1VisualLock,
  buildCharacterPhysicalLock,
  setApiKey
} from './services/geminiService';
import { AppState, EngineeredScene, ScriptScene, VocalGear } from './types';

export default function App() {
  const [state, setState] = useState<AppState>({
    step: 'upload',
    referenceVideo: null,
    targetCharacterImages: [],
    newScript: '',
    referenceAnalysis: null,
    scriptSegmentation: null,
    selectedSceneIndex: null,
    sceneProcessing: 'idle',
    sceneProcessingStatus: '',
    currentPrompt: null,
    completedScenes: [],
    isRefining: false,
    refinementFeedback: '',
    showRefineInput: false,
    learnedPreferences: null,
    characterPhysicalLock: null,
    error: null,
    processingStatus: ''
  });

  const [copied, setCopied]           = useState(false);
  const [apiKey, setApiKeyState]      = useState('');
  const [isFreeTierApi, setIsFreeTierApi] = useState(false);
  const [isKeySet, setIsKeySet]       = useState(false);
  const [charDragOver, setCharDragOver]   = useState(false);
  const [scene1VisualLock, setScene1VisualLock] = useState<string | null>(null);

  const videoInputRef   = useRef<HTMLInputElement>(null);
  const charInputRef    = useRef<HTMLInputElement>(null);

  // Thumbnail object-URL cache (avoid re-creating on every render)
  const thumbCache = useRef<Map<string, string>>(new Map());
  const getThumb = (file: File) => {
    if (!thumbCache.current.has(file.name + file.size)) {
      thumbCache.current.set(file.name + file.size, URL.createObjectURL(file));
    }
    return thumbCache.current.get(file.name + file.size)!;
  };

  useEffect(() => {
    const stored = localStorage.getItem('gemini_api_key');
    const storedFreeTier = localStorage.getItem('gemini_is_free_tier') === 'true';
    if (stored) {
      setApiKey(stored, storedFreeTier);
      setApiKeyState(stored);
      setIsFreeTierApi(storedFreeTier);
      setIsKeySet(true);
    }
  }, []);

  const handleSaveKey = () => {
    if (!apiKey.trim()) return;
    setApiKey(apiKey, isFreeTierApi);
    localStorage.setItem('gemini_api_key', apiKey);
    localStorage.setItem('gemini_is_free_tier', String(isFreeTierApi));
    setIsKeySet(true);
  };

  const handleError = (err: any) => {
    // Log the full error to console for easier debugging of "unexpected" issues
    console.error("AL-NOKHBA ERROR DETECTED:", err);
    setState(s => ({
      ...s,
      error: err?.message ? `Error: ${err.message}` : typeof err === 'string' ? `Error: ${err}` : 'An unexpected error occurred. Check browser console.',
      processingStatus: '',
      sceneProcessing: s.sceneProcessing === 'engineering' ? 'idle' : s.sceneProcessing,
    }));
  };

  // ── Character image helpers ─────────────────────────────────────────────
  const addCharacterImages = useCallback((files: FileList | File[]) => {
    const incoming = Array.from(files).filter(f => f.type.startsWith('image/'));
    setState(s => ({
      ...s,
      targetCharacterImages: [...s.targetCharacterImages, ...incoming].slice(0, 5)
    }));
  }, []);

  const removeCharacterImage = (idx: number) => {
    setState(s => ({
      ...s,
      targetCharacterImages: s.targetCharacterImages.filter((_, i) => i !== idx)
    }));
  };

  const handleCharDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setCharDragOver(false);
    addCharacterImages(e.dataTransfer.files);
  };

  // ── Main flow ────────────────────────────────────────────────────────────
  const handleBeginAnalysis = async () => {
    if (!state.referenceVideo || state.targetCharacterImages.length === 0 || !state.newScript.trim()) return;
    try {
      setState(s => ({ ...s, step: 'analyzing-reference', processingStatus: 'Phase 1/2 — Extracting Performance DNA...', error: null }));
      const analysis = await analyzeReferenceVideo(state.referenceVideo);

      setState(s => ({ ...s, step: 'segmenting', processingStatus: 'Phase 2/2 — Architecting Scene Sequence...', referenceAnalysis: analysis }));
      const segmentation = await segmentScript(state.newScript, analysis);

      setState(s => ({ ...s, step: 'scenes', scriptSegmentation: segmentation }));
    } catch (err) {
      handleError(err);
      setState(s => ({ ...s, step: 'upload' }));
    }
  };

  const selectScene = (index: number) => {
    if (!state.scriptSegmentation) return;
    const scene    = state.scriptSegmentation.scenes[index];
    const existing = state.completedScenes.find(c => c.scene_number === scene.scene_number);
    setState(s => ({
      ...s,
      selectedSceneIndex: index,
      currentPrompt: existing ? existing.veo_prompt : null,
      sceneProcessing: existing ? 'complete' : 'idle',
      sceneProcessingStatus: '',
      error: null,
    }));
  };

  const handleEngineerScene = async () => {
    if (state.selectedSceneIndex === null || !state.scriptSegmentation || !state.referenceAnalysis) return;
    const scene = state.scriptSegmentation.scenes[state.selectedSceneIndex];
    try {
      const seg = state.scriptSegmentation;
      setState(s => ({ ...s, sceneProcessing: 'engineering', sceneProcessingStatus: 'Agent Pass A — Scene Intelligence Extraction...', error: null }));

      // Build character physical lock once per session (first scene only)
      let physicalLock = state.characterPhysicalLock;
      if (!physicalLock && state.targetCharacterImages.length > 0) {
        setState(s => ({ ...s, sceneProcessingStatus: 'Building character identity lock from photos...' }));
        try {
          physicalLock = await buildCharacterPhysicalLock(state.targetCharacterImages);
          setState(s => ({ ...s, characterPhysicalLock: physicalLock }));
        } catch {
          // Non-fatal — proceed without the lock; photos are still attached
          physicalLock = null;
        }
      }

      const finalPrompt = await runVeoPromptAgent(
        scene,
        state.referenceAnalysis,
        state.targetCharacterImages,
        state.completedScenes,
        seg?.scenes,
        seg ? {
          narrative_arc:    seg.narrative_arc,
          total_scenes:     seg.total_scenes,
          directing_vision: seg.directing_vision,
        } : undefined,
        state.learnedPreferences || undefined,
        scene.scene_number > 1 ? (scene1VisualLock || undefined) : undefined,
        (msg) => setState(s => ({ ...s, sceneProcessingStatus: msg })),
        physicalLock || undefined,
      );

      // Extract Scene 1 visual constants to lock into all subsequent scenes
      if (scene.scene_number === 1 && !scene1VisualLock) {
        const lock = buildScene1VisualLock(finalPrompt);
        setScene1VisualLock(lock);
      }

      const engineered: EngineeredScene = {
        scene_number:    scene.scene_number,
        scene_title:     scene.title,
        role:            scene.role,
        duration_seconds: scene.duration_seconds,
        veo_prompt:      finalPrompt,
        timestamp:       new Date().toISOString(),
      };
      setState(s => ({
        ...s, sceneProcessing: 'complete', currentPrompt: finalPrompt,
        completedScenes: [...s.completedScenes.filter(c => c.scene_number !== scene.scene_number), engineered],
      }));
    } catch (err) { handleError(err); }
  };

  const handleRefineScene = async () => {
    if (state.selectedSceneIndex === null || !state.scriptSegmentation || !state.currentPrompt) return;
    if (!state.refinementFeedback.trim()) return;
    const scene = state.scriptSegmentation.scenes[state.selectedSceneIndex];
    try {
      setState(s => ({ ...s, isRefining: true, error: null }));
      const refined = await refineScenePrompt(state.currentPrompt, scene, state.refinementFeedback);
      const updated = state.completedScenes.find(c => c.scene_number === scene.scene_number);
      const updatedScene = updated
        ? { ...updated, veo_prompt: refined, refinement_notes: state.refinementFeedback }
        : null;
      setState(s => ({
        ...s,
        isRefining: false,
        showRefineInput: false,
        refinementFeedback: '',
        currentPrompt: refined,
        completedScenes: updatedScene
          ? [...s.completedScenes.filter(c => c.scene_number !== scene.scene_number), updatedScene]
          : s.completedScenes
      }));
    } catch (err) { handleError(err); setState(s => ({ ...s, isRefining: false })); }
  };

  const handleMergeNextScene = () => {
    if (state.selectedSceneIndex === null || !state.scriptSegmentation) return;
    const idx    = state.selectedSceneIndex;
    const scenes = [...state.scriptSegmentation.scenes];
    if (idx >= scenes.length - 1) return;
    const cur = scenes[idx], nxt = scenes[idx + 1];

    // Smart merge validation
    const combinedWords  = (cur.word_count || 0) + (nxt.word_count || 0);
    const energyDelta    = Math.abs(cur.energy_level - nxt.energy_level);
    const warnings: string[] = [];
    if (combinedWords > 20) warnings.push(`Combined word count (${combinedWords}) exceeds 20 — VEO may struggle with timing.`);
    if (energyDelta >= 3)   warnings.push(`Energy shift ${cur.energy_level}→${nxt.energy_level} — merged scene may feel inconsistent.`);
    if (warnings.length > 0 && !window.confirm(`⚠ Merge Warning:\n\n${warnings.join('\n')}\n\nProceed anyway?`)) return;

    const merged: ScriptScene = {
      ...cur,
      title:            `${cur.title} + ${nxt.title}`,
      duration_seconds: cur.duration_seconds + nxt.duration_seconds,
      script_text:      `${cur.script_text} ${nxt.script_text}`,
      acting_blueprint: {
        ...cur.acting_blueprint,
        intention:         `${cur.acting_blueprint.intention} → ${nxt.acting_blueprint.intention}`,
        subtext:           `${cur.acting_blueprint.subtext} → ${nxt.acting_blueprint.subtext}`,
        mapped_gestures:   Array.from(new Set([...cur.acting_blueprint.mapped_gestures,   ...nxt.acting_blueprint.mapped_gestures])),
        mapped_mannerisms: Array.from(new Set([...cur.acting_blueprint.mapped_mannerisms, ...nxt.acting_blueprint.mapped_mannerisms])),
        pause_map:         [...cur.acting_blueprint.pause_map,   ...nxt.acting_blueprint.pause_map],
        emphasis_words:    [...cur.acting_blueprint.emphasis_words, ...nxt.acting_blueprint.emphasis_words]
      },
      recommended_outframe: nxt.recommended_outframe,
      continuity: { enters_from: cur.continuity.enters_from, exits_to: nxt.continuity.exits_to }
    };
    scenes.splice(idx, 2, merged);
    for (let i = idx + 1; i < scenes.length; i++) scenes[i].scene_number = i + 1;
    setState(s => ({
      ...s,
      scriptSegmentation: { ...s.scriptSegmentation!, scenes, total_scenes: scenes.length },
      completedScenes: s.completedScenes
        .filter(c => c.scene_number !== cur.scene_number && c.scene_number !== nxt.scene_number)
        .map(c => c.scene_number > nxt.scene_number ? { ...c, scene_number: c.scene_number - 1 } : c),
      selectedSceneIndex: null, sceneProcessing: 'idle', currentPrompt: null
    }));
  };

  // ── Star rating & energy calibration ─────────────────────────────────────
  const handleRateScene = (sceneNumber: number, stars: number) => {
    setState(s => {
      const updatedCompleted = s.completedScenes.map(c =>
        c.scene_number === sceneNumber ? { ...c, rating: stars } : c
      );
      // Learn from 4-5 star scenes once we have 2+ of them
      const ratedHigh = updatedCompleted.filter(c => (c.rating || 0) >= 4);
      let newPrefs = s.learnedPreferences;
      if (ratedHigh.length >= 2 && s.scriptSegmentation) {
        const highNums = new Set(ratedHigh.map(c => c.scene_number));
        const highScenes = s.scriptSegmentation.scenes.filter(sc => highNums.has(sc.scene_number));
        // Most common vocal gear among top-rated
        const gearCounts: Record<number, number> = {};
        highScenes.forEach(sc => {
          const g = (sc.acting_blueprint as any).vocal_gear as number | undefined;
          if (g) gearCounts[g] = (gearCounts[g] || 0) + 1;
        });
        const preferredGear = Object.keys(gearCounts).sort((a, b) => gearCounts[+b] - gearCounts[+a])[0];
        // Most common focal length category
        const focalCounts: Record<string, number> = {};
        highScenes.forEach(sc => {
          const f = (sc.acting_blueprint as any).focal_length as string | undefined;
          if (f) {
            const key = f.includes('85') ? '85mm+' : (f.includes('35') || f.includes('50')) ? '35-50mm' : '16-24mm';
            focalCounts[key] = (focalCounts[key] || 0) + 1;
          }
        });
        const preferredFocal = Object.keys(focalCounts).sort((a, b) => focalCounts[b] - focalCounts[a])[0];
        const avgEnergy = highScenes.length > 0
          ? Math.round(highScenes.reduce((a, sc) => a + sc.energy_level, 0) / highScenes.length)
          : undefined;
        newPrefs = {
          gear:   preferredGear ? parseInt(preferredGear) : undefined,
          focal:  preferredFocal,
          energy: avgEnergy,
        };
      }
      return { ...s, completedScenes: updatedCompleted, learnedPreferences: newPrefs };
    });
  };

  const copyPrompt = () => {
    if (!state.currentPrompt) return;
    navigator.clipboard.writeText(state.currentPrompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  // Vocal gear badge
  const vocalGearBadge = (gear?: VocalGear) => {
    if (!gear) return null;
    const map: Record<number, { label: string; cls: string }> = {
      4: { label: 'G4', cls: 'bg-red-500/15 text-red-400 border-red-500/30' },
      3: { label: 'G3', cls: 'bg-orange-500/15 text-orange-400 border-orange-500/30' },
      2: { label: 'G2', cls: 'bg-blue-500/15 text-blue-400 border-blue-500/30' },
      1: { label: 'G1', cls: 'bg-purple-500/15 text-purple-400 border-purple-500/30' },
    };
    const d = map[gear];
    return d ? <span className={`text-[7px] font-mono font-bold px-1.5 py-0.5 rounded border ${d.cls}`} title={`Vocal Gear ${gear}`}>{d.label}</span> : null;
  };

  // Retention target badge
  const retentionBadge = (pct?: number) => {
    if (!pct) return null;
    const cls = pct >= 85 ? 'text-emerald-400' : pct >= 70 ? 'text-gold/80' : 'text-slate-500';
    return <span className={`text-[7px] font-mono ${cls}`} title="Retention target">{pct}%</span>;
  };

  // Lip sync risk badge
  const lipSyncBadge = (risk?: 'low' | 'medium' | 'high') => {
    if (!risk) return null;
    const map = {
      low:    { label: 'LS✓', cls: 'bg-emerald-500/10 text-emerald-600 border-emerald-500/20' },
      medium: { label: 'LS~', cls: 'bg-amber-500/10 text-amber-500 border-amber-500/20' },
      high:   { label: 'LS!', cls: 'bg-red-500/10 text-red-500 border-red-500/20' },
    };
    const d = map[risk];
    return <span className={`text-[7px] font-mono font-bold px-1 py-0.5 rounded border ${d.cls}`} title={`Lip sync risk: ${risk}`}>{d.label}</span>;
  };

  // Word count validation — returns warning if word count exceeds role-based WPM limit
  const wordCountOk = (scene: ScriptScene): boolean => {
    const extendedRoles = ['Framework','Action Framework','Storytelling','Case Study'];
    const midRoles = ['Perspective Shift','Objection Handler','Closing'];
    const max = extendedRoles.includes(scene.role) ? 16 : midRoles.includes(scene.role) ? 13 : 12;
    return (scene.word_count || 0) <= max;
  };

  const roleColor = (role: string) => ({
    // Core roles
    'Hook':              'bg-red-500/15 text-red-400 border-red-500/25',
    'Pattern Interrupt': 'bg-amber-500/15 text-amber-400 border-amber-500/25',
    'Value Delivery':    'bg-blue-500/15 text-blue-400 border-blue-500/25',
    'Social Proof':      'bg-emerald-500/15 text-emerald-400 border-emerald-500/25',
    'Bridge':            'bg-slate-500/15 text-slate-400 border-slate-500/25',
    'Call to Action':    'bg-green-500/15 text-green-400 border-green-500/25',
    'Storytelling':      'bg-purple-500/15 text-purple-400 border-purple-500/25',
    'Demonstration':     'bg-cyan-500/15 text-cyan-400 border-cyan-500/25',
    'Objection Handler': 'bg-rose-500/15 text-rose-400 border-rose-500/25',
    'Open Loop':         'bg-orange-500/15 text-orange-400 border-orange-500/25',
    'Closing':           'bg-indigo-500/15 text-indigo-400 border-indigo-500/25',
    // YouTube Thought Leadership roles
    'Insight Reveal':    'bg-violet-500/15 text-violet-400 border-violet-500/25',
    'Framework':         'bg-sky-500/15 text-sky-400 border-sky-500/25',
    'Case Study':        'bg-teal-500/15 text-teal-400 border-teal-500/25',
    'Market Intelligence':'bg-lime-600/15 text-lime-400 border-lime-600/25',
    'Perspective Shift': 'bg-fuchsia-500/15 text-fuchsia-400 border-fuchsia-500/25',
    'Action Framework':  'bg-emerald-600/15 text-emerald-300 border-emerald-600/25',
  }[role] || 'bg-slate-500/15 text-slate-400 border-slate-500/25');

  // Role-aware duration cap (mirrors geminiService ROLE_MAX_SECONDS)
  const roleMaxSeconds = (role: string) => ({
    'Framework': 12, 'Action Framework': 12, 'Storytelling': 12, 'Case Study': 12,
    'Perspective Shift': 10, 'Closing': 10, 'Objection Handler': 10,
  } as Record<string, number>)[role] ?? 8;

  // Export all completed prompts as a single .txt file
  const exportAllPrompts = () => {
    const sorted = [...state.completedScenes].sort((a, b) => a.scene_number - b.scene_number);
    const text = sorted.map(s =>
      `${'═'.repeat(60)}\nSCENE ${String(s.scene_number).padStart(2, '0')} — ${s.scene_title}  [${s.role}]  ${s.duration_seconds}s\n${'═'.repeat(60)}\n\n${s.veo_prompt}`
    ).join('\n\n\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = 'veo-prompts-all-scenes.txt';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const canBegin = state.referenceVideo && state.targetCharacterImages.length > 0 && state.newScript.trim().length > 0;

  // ═══════════════════════════════════════════════════════════════
  // API KEY GATE
  // ═══════════════════════════════════════════════════════════════
  if (!isKeySet) return (
    <div className="min-h-screen flex items-center justify-center bg-[#020617] p-6">
      <div className="max-w-md w-full space-y-7 bg-gradient-to-b from-slate-900/80 to-slate-950/80 p-10 rounded-2xl border border-white/[0.07] shadow-[0_0_60px_rgba(0,0,0,0.6)] backdrop-blur-sm">
        <div className="text-center space-y-2">
          <div className="text-[10px] tracking-[0.35em] text-gold/50 uppercase font-mono">AL-NOKHBA — v4.0</div>
          <h2 className="text-3xl font-serif italic text-white text-glow mt-2">Gemini API Key</h2>
          <p className="text-slate-500 text-sm leading-relaxed pt-1">
            Access the Elite VEO 3.1 Scene Director powered by Gemini 3.1 Pro.
          </p>
        </div>
        <div className="space-y-3">
          <input
            type="password"
            value={apiKey}
            onChange={e => setApiKeyState(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSaveKey()}
            placeholder="AIza..."
            className="w-full bg-black/60 border border-white/[0.08] rounded-xl px-4 py-3.5 text-white placeholder:text-slate-700 focus:border-gold/30 focus:outline-none font-mono text-sm tracking-wider transition-colors"
          />
          <label className="flex items-center gap-3 px-1 py-1 cursor-pointer">
            <input
              type="checkbox"
              checked={isFreeTierApi}
              onChange={e => setIsFreeTierApi(e.target.checked)}
              className="w-4 h-4 rounded border-white/[0.2] bg-black/60 text-gold focus:ring-gold/50 focus:ring-offset-0"
            />
            <span className="text-sm text-slate-400 font-mono">This is a Free Tier API Key</span>
          </label>
          <button
            onClick={handleSaveKey}
            disabled={!apiKey.trim()}
            className="w-full py-3.5 bg-gold text-black font-bold rounded-xl hover:bg-yellow-500 transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed shadow-[0_4px_30px_rgba(202,138,4,0.25)] hover:shadow-[0_4px_40px_rgba(202,138,4,0.4)] tracking-wider text-sm"
          >
            AUTHENTICATE
          </button>
        </div>
        <p className="text-center">
          <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noreferrer"
            className="text-xs text-slate-700 hover:text-slate-400 transition-colors underline underline-offset-2">
            Get a free API key at Google AI Studio
          </a>
        </p>
      </div>
    </div>
  );

  // ═══════════════════════════════════════════════════════════════
  // MAIN APP
  // ═══════════════════════════════════════════════════════════════
  return (
    <div className="min-h-screen flex flex-col font-sans selection:bg-gold/25 relative">

      {/* Error banner */}
      {state.error && (
        <div className="fixed top-0 inset-x-0 z-50 bg-red-950/95 border-b border-red-800/40 backdrop-blur-md px-6 py-3 flex items-center justify-between text-sm">
          <span className="flex items-center gap-2 text-red-200">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400 shrink-0" />
            {state.error}
          </span>
          <button onClick={() => setState(s => ({ ...s, error: null }))} className="text-red-400 hover:text-white transition-colors text-lg ml-6 shrink-0">&times;</button>
        </div>
      )}

      {/* ╔══════════════════════════════════════════════════╗
          ║  VIEW 1 — UPLOAD                                 ║
          ╚══════════════════════════════════════════════════╝ */}
      {state.step === 'upload' && (
        <div className="flex-1 flex items-center justify-center p-6 overflow-y-auto bg-[radial-gradient(ellipse_120%_60%_at_50%_-10%,rgba(202,138,4,0.04)_0%,transparent_70%)]">
          <div className="max-w-3xl w-full space-y-7 py-14">

            {/* Hero */}
            <div className="text-center space-y-3">
              <div className="text-[9px] tracking-[0.45em] text-gold/35 uppercase font-mono">AL-NOKHBA — Elite VSL Production</div>
              <h1 className="font-serif italic text-[3.5rem] leading-tight text-white text-glow">VEO 3.1 Scene Director</h1>
              <p className="text-[10px] tracking-[0.3em] text-slate-700 uppercase">Your script. Their presence. Zero compromise.</p>
            </div>

            {/* ── Row 1: Reference Video + Character Images ── */}
            <div className="grid grid-cols-5 gap-5">

              {/* Reference Video — 2/5 */}
              <div
                onClick={() => videoInputRef.current?.click()}
                className={`col-span-2 border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer group transition-all duration-300 ${state.referenceVideo ? 'border-gold/40 bg-gold/[0.03]' : 'border-slate-800 hover:border-slate-700 hover:bg-white/[0.01]'}`}
              >
                <input ref={videoInputRef} type="file" accept="video/*" className="hidden"
                  onChange={e => setState(s => ({ ...s, referenceVideo: e.target.files?.[0] || null }))} />
                <div className={`w-9 h-9 mx-auto mb-3 transition-colors ${state.referenceVideo ? 'text-gold' : 'text-slate-700 group-hover:text-slate-500'}`}>
                  <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                {state.referenceVideo ? (
                  <div>
                    <p className="text-white text-sm font-medium truncate px-2">{state.referenceVideo.name}</p>
                    <p className="text-[9px] text-gold uppercase tracking-widest mt-1.5 font-mono">Reference Loaded</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-slate-400 group-hover:text-slate-200 transition-colors text-sm font-medium">Reference Video</p>
                    <p className="text-slate-700 text-xs mt-1">Performer to capture</p>
                  </div>
                )}
              </div>

              {/* Character Images — 3/5 */}
              <div className="col-span-3">
                <input ref={charInputRef} type="file" accept="image/*" multiple className="hidden"
                  onChange={e => e.target.files && addCharacterImages(e.target.files)} />

                {state.targetCharacterImages.length === 0 ? (
                  /* Empty state — drag & drop zone */
                  <div
                    onClick={() => charInputRef.current?.click()}
                    onDragOver={e => { e.preventDefault(); setCharDragOver(true); }}
                    onDragLeave={() => setCharDragOver(false)}
                    onDrop={handleCharDrop}
                    className={`h-full min-h-[140px] border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center group ${charDragOver ? 'border-gold/60 bg-gold/[0.05]' : 'border-slate-800 hover:border-slate-700 hover:bg-white/[0.01]'}`}
                  >
                    <div className="w-9 h-9 text-slate-700 group-hover:text-slate-500 transition-colors mb-3">
                      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <p className="text-slate-400 group-hover:text-slate-200 transition-colors text-sm font-medium">Upload Character Photos</p>
                    <p className="text-slate-700 text-xs mt-1">Drag & drop or click — up to 5 photos</p>
                    <p className="text-[9px] text-slate-800 font-mono mt-2 tracking-wider uppercase">More photos = better identity fidelity</p>
                  </div>
                ) : (
                  /* Thumbnail grid */
                  <div
                    onDragOver={e => { e.preventDefault(); setCharDragOver(true); }}
                    onDragLeave={() => setCharDragOver(false)}
                    onDrop={handleCharDrop}
                    className={`border-2 rounded-2xl p-4 transition-all duration-200 ${charDragOver ? 'border-gold/50 bg-gold/[0.04]' : 'border-slate-800/80 bg-slate-950/30'}`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-[9px] text-gold/70 uppercase tracking-widest font-mono">
                          {state.targetCharacterImages.length} photo{state.targetCharacterImages.length > 1 ? 's' : ''} — Identity Source
                        </span>
                        {state.targetCharacterImages.length >= 3 && (
                          <span className="text-[8px] text-emerald-600 font-mono uppercase tracking-wider border border-emerald-800/40 px-1.5 py-0.5 rounded">
                            Optimal
                          </span>
                        )}
                      </div>
                      {state.targetCharacterImages.length < 5 && (
                        <button
                          onClick={() => charInputRef.current?.click()}
                          className="text-[10px] text-slate-500 hover:text-gold transition-colors font-mono flex items-center gap-1"
                        >
                          <span className="text-base leading-none">+</span> Add more
                        </button>
                      )}
                    </div>

                    <div className="grid grid-cols-5 gap-2">
                      {state.targetCharacterImages.map((img, i) => (
                        <div key={i} className="relative group/thumb aspect-square">
                          <img
                            src={getThumb(img)}
                            alt={`Character ${i + 1}`}
                            className="w-full h-full object-cover rounded-lg border border-white/[0.06]"
                          />
                          {/* Photo number */}
                          <div className="absolute bottom-1 left-1 text-[8px] font-mono text-white/50 bg-black/60 px-1 rounded leading-tight">
                            #{i + 1}
                          </div>
                          {/* Remove button */}
                          <button
                            onClick={() => removeCharacterImage(i)}
                            className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-red-500 rounded-full text-white text-[9px] font-bold flex items-center justify-center opacity-0 group-hover/thumb:opacity-100 transition-opacity shadow-lg"
                          >
                            ×
                          </button>
                        </div>
                      ))}

                      {/* Add slot if < 5 */}
                      {state.targetCharacterImages.length < 5 && (
                        <button
                          onClick={() => charInputRef.current?.click()}
                          className="aspect-square rounded-lg border-2 border-dashed border-slate-800 hover:border-slate-600 hover:bg-white/[0.02] transition-all flex items-center justify-center text-slate-700 hover:text-slate-500 text-lg"
                        >
                          +
                        </button>
                      )}
                    </div>

                    {state.targetCharacterImages.length === 1 && (
                      <p className="text-[9px] text-slate-700 font-mono mt-2.5 tracking-wider">
                        TIP: Add 2–4 more photos (front, 3/4, side) for maximum identity accuracy
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* ── Script Textarea ── */}
            <div className="relative group">
              <div className="absolute -inset-px rounded-2xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500 pointer-events-none" style={{ background: 'linear-gradient(135deg,rgba(202,138,4,0.08),transparent,rgba(202,138,4,0.05))' }} />
              <textarea
                value={state.newScript}
                onChange={e => setState(s => ({ ...s, newScript: e.target.value }))}
                placeholder="Paste your video script here — raw dialogue or Elite Blueprint format [SCENE START], [VISUAL], [ACTING]..."
                className="relative w-full bg-[#080e1d] text-slate-300 p-6 rounded-xl border border-white/[0.06] focus:border-gold/20 focus:ring-1 focus:ring-gold/10 outline-none min-h-[200px] font-mono text-sm leading-relaxed resize-y placeholder:text-slate-700 transition-colors"
              />
              <div className="absolute bottom-3.5 right-4 flex items-center gap-3 pointer-events-none">
                <span className="text-[9px] text-slate-700 font-mono">
                  {state.newScript.split(/\s+/).filter(Boolean).length} words
                </span>
              </div>
            </div>

            {/* ── Begin Button ── */}
            <button
              disabled={!canBegin}
              onClick={handleBeginAnalysis}
              className="w-full py-4 bg-white text-black text-lg font-serif italic rounded-xl hover:bg-slate-50 disabled:opacity-25 disabled:cursor-not-allowed transition-all duration-300 shadow-[0_0_40px_rgba(255,255,255,0.04)] hover:shadow-[0_0_50px_rgba(255,255,255,0.12)] tracking-wide"
            >
              Begin Performance DNA Extraction
            </button>

            <p className="text-center text-[9px] text-slate-800 font-mono tracking-[0.3em] uppercase">
              Gemini 3.1 Pro — High Thinking Mode — YouTube Directing &amp; Acting Bible Agents — Photo-Anchored Identity Lock
            </p>
          </div>
        </div>
      )}

      {/* ╔══════════════════════════════════════════════════╗
          ║  VIEW 2 — LOADING                                ║
          ╚══════════════════════════════════════════════════╝ */}
      {(state.step === 'analyzing-reference' || state.step === 'segmenting') && (
        <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-10 bg-[#020617]">
          <div className="relative w-28 h-28">
            <div className="absolute inset-0 border-[3px] border-t-gold border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" />
            <div className="absolute inset-3 border-[2px] border-t-gold/30 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" style={{ animationDuration: '1.8s', animationDirection: 'reverse' }} />
            <div className="absolute inset-6 border-[1.5px] border-t-gold/10 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" style={{ animationDuration: '3.2s' }} />
          </div>
          <div className="text-center space-y-5">
            <h2 className="font-serif italic text-4xl text-white">
              {state.step === 'analyzing-reference' ? 'Extracting Performance DNA' : 'Architecting Scene Sequence'}
            </h2>
            <p className="text-gold/40 text-xs font-mono tracking-[0.25em] uppercase">{state.processingStatus}</p>
            <div className="flex items-center justify-center gap-3 text-xs font-mono mt-2">
              <span className={`flex items-center gap-1.5 ${state.step === 'analyzing-reference' ? 'text-gold' : 'text-emerald-500'}`}>
                <span>{state.step === 'analyzing-reference' ? '●' : '✓'}</span>
                Phase 1 — DNA Extraction
              </span>
              <span className="text-slate-800">──</span>
              <span className={`flex items-center gap-1.5 ${state.step === 'segmenting' ? 'text-gold' : 'text-slate-800'}`}>
                <span>{state.step === 'segmenting' ? '●' : '○'}</span>
                Phase 2 — Bible-Informed Scene Architecture
              </span>
            </div>
            <div className="flex items-center justify-center gap-4 mt-4 text-[8px] font-mono text-slate-800 uppercase tracking-wider">
              <span>📖 Directing Bible Active</span>
              <span>·</span>
              <span>🎭 Acting Bible Active</span>
              <span>·</span>
              <span>Retention Curves</span>
              <span>·</span>
              <span>Vocal Gear System</span>
            </div>
            <p className="text-slate-800 text-xs font-mono mt-3">This may take 60–120 seconds — high thinking mode engaged</p>
          </div>
        </div>
      )}

      {/* ╔══════════════════════════════════════════════════╗
          ║  VIEW 3 — SCENES                                 ║
          ╚══════════════════════════════════════════════════╝ */}
      {state.step === 'scenes' && state.scriptSegmentation && (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-[#020617]">

          {/* Header */}
          <div className="border-b border-white/[0.05] px-6 py-3 flex items-center justify-between shrink-0 bg-[#020617]/95 backdrop-blur-md z-20">
            <div className="flex items-center gap-4">
              <h1 className="font-serif italic text-xl text-white text-glow">AL-NOKHBA</h1>
              <span className="text-slate-800">|</span>
              <span className="text-slate-700 text-[10px] tracking-[0.2em] uppercase font-mono">Scene Director v4.0</span>
            </div>
            <div className="flex items-center gap-5">
              {/* Character identity pills */}
              <div className="flex items-center gap-1.5">
                {state.targetCharacterImages.slice(0, 3).map((img, i) => (
                  <img key={i} src={getThumb(img)} alt="" className="w-6 h-6 rounded-full object-cover border border-gold/20 ring-1 ring-black" />
                ))}
                {state.targetCharacterImages.length > 3 && (
                  <span className="w-6 h-6 rounded-full bg-slate-800 border border-gold/20 text-[8px] text-slate-400 flex items-center justify-center font-mono">
                    +{state.targetCharacterImages.length - 3}
                  </span>
                )}
                <span className="text-[9px] text-slate-600 font-mono ml-1">Identity</span>
              </div>
              <div className="flex items-center gap-2 text-[10px]">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_5px_rgba(16,185,129,0.5)]" />
                <span className="text-slate-600 font-mono">DNA Active</span>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="text-slate-700 text-xs font-mono">{state.completedScenes.length}/{state.scriptSegmentation.scenes.length}</span>
                <div className="w-32 h-px bg-slate-900 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gold shadow-[0_0_6px_rgba(202,138,4,0.4)] transition-all duration-700"
                    style={{ width: `${(state.completedScenes.length / state.scriptSegmentation.scenes.length) * 100}%` }}
                  />
                </div>
              </div>
              {state.completedScenes.length > 0 && (
                <button
                  onClick={exportAllPrompts}
                  title={`Export ${state.completedScenes.length} completed prompt${state.completedScenes.length > 1 ? 's' : ''}`}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-white/[0.05] hover:border-gold/20 rounded-lg text-[9px] font-mono text-slate-500 hover:text-gold/80 transition-all duration-200"
                >
                  <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                  Export {state.completedScenes.length === state.scriptSegmentation.scenes.length ? 'All' : state.completedScenes.length}
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 flex overflow-hidden">

            {/* ── Sidebar ───────────────────────────────────────── */}
            <div className="w-[380px] border-r border-white/[0.04] flex flex-col bg-[#010510] shrink-0 overflow-hidden">
              {/* Dynamic Fatigue Mapping UI */}
              <div className="p-4 border-b border-white/[0.04] shrink-0">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[9px] text-slate-500 font-mono uppercase tracking-widest">Energy Arc Map</span>
                  {(() => {
                    const scenes = state.scriptSegmentation.scenes;
                    const lowEnergyStreak = scenes.some((s, i) => i < scenes.length - 2 && s.energy_level <= 4 && scenes[i+1].energy_level <= 4 && scenes[i+2].energy_level <= 4);
                    return lowEnergyStreak ? <span className="text-[8px] text-red-400 font-mono bg-red-500/10 px-1.5 py-0.5 rounded border border-red-500/20" title="Viewer fatigue risk: 3+ consecutive low energy scenes">⚠ Fatigue Risk</span> : null;
                  })()}
                </div>
                <div className="flex items-end gap-[2px] h-10 w-full relative">
                  {state.scriptSegmentation.scenes.map((s, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-slate-800 rounded-t-sm transition-all hover:bg-gold/50 cursor-pointer group"
                      style={{ height: `${Math.max(10, (s.energy_level / 10) * 100)}%`, backgroundColor: state.selectedSceneIndex === i ? 'rgba(202,138,4,0.6)' : undefined }}
                      onClick={() => selectScene(i)}
                    >
                      <div className="opacity-0 group-hover:opacity-100 absolute -top-5 left-1/2 -translate-x-1/2 bg-black/80 text-white text-[8px] px-1.5 py-0.5 rounded font-mono pointer-events-none whitespace-nowrap">
                        Scene {s.scene_number}: E{s.energy_level}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex-1 overflow-y-auto p-3 space-y-1.5 custom-scrollbar">
              {state.scriptSegmentation.scenes.map((scene, idx) => {
                const sel  = state.selectedSceneIndex === idx;
                const done = state.completedScenes.some(c => c.scene_number === scene.scene_number);
                return (
                  <button key={idx} onClick={() => selectScene(idx)}
                    className={`w-full text-left p-4 rounded-xl border transition-all duration-200 group relative overflow-hidden ${sel ? 'border-gold/25 bg-gold/[0.03]' : 'border-white/[0.03] hover:border-white/[0.08] hover:bg-white/[0.015]'}`}
                  >
                    {sel && <div className="absolute inset-y-0 left-0 w-0.5 bg-gold/50" />}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-1.5">
                        <span className={`font-mono text-[10px] font-bold ${sel ? 'text-gold/70' : 'text-slate-700'}`}>#{String(scene.scene_number).padStart(2, '0')}</span>
                        <span className={`text-[8px] uppercase tracking-wider px-1.5 py-0.5 rounded-full border ${roleColor(scene.role)}`}>{scene.role}</span>
                        {scene.is_pattern_interrupt && <span className="text-[7px] text-amber-400 font-mono border border-amber-500/25 px-1 py-0.5 rounded bg-amber-500/8" title="Pattern Interrupt">⚡PI</span>}
                      </div>
                      <div className="flex items-center gap-1.5">
                        {done && <span className="text-emerald-500 text-xs">✓</span>}
                        {vocalGearBadge(scene.acting_blueprint?.vocal_gear)}
                        {retentionBadge(scene.retention_target_percent)}
                        {lipSyncBadge(scene.lip_sync_risk)}
                        {!wordCountOk(scene) && <span className="text-[7px] font-mono text-red-500 border border-red-500/20 px-1 py-0.5 rounded bg-red-500/8" title="Word count exceeds WPM limit">W!</span>}
                        <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${scene.duration_seconds > roleMaxSeconds(scene.role) ? 'text-red-400 bg-red-500/10' : scene.duration_seconds <= 5.5 ? 'text-amber-500/80 bg-amber-500/8' : 'text-slate-700'}`}>{scene.duration_seconds}s</span>
                      </div>
                    </div>
                    <p className={`font-serif italic text-base leading-snug mb-1.5 transition-colors ${sel ? 'text-white' : 'text-slate-400 group-hover:text-slate-300'}`}>{scene.title}</p>
                    <p className="text-[9px] text-slate-800 font-mono truncate">"{scene.script_text.substring(0, 52)}..."</p>
                    {/* Energy bar */}
                    <div className="mt-2.5 flex items-center gap-2">
                      <div className="flex-1 h-px bg-slate-900/80">
                        <div className={`h-full ${scene.energy_level >= 7 ? 'bg-gold/60' : scene.energy_level >= 4 ? 'bg-slate-600' : 'bg-slate-800'}`} style={{ width: `${scene.energy_level * 10}%` }} />
                      </div>
                      <span className="text-[8px] text-slate-800 font-mono">{scene.energy_level}/10</span>
                    </div>
                  </button>
                );
              })}
              </div>
            </div>

            {/* ── Workspace ─────────────────────────────────────── */}
            <div className="flex-1 overflow-y-auto custom-scrollbar bg-[#020617]">
              {state.selectedSceneIndex === null ? (

                <div className="h-full overflow-y-auto custom-scrollbar">
                  <div className="max-w-3xl mx-auto px-8 py-10 space-y-8">

                    {/* Directing Vision */}
                    {state.scriptSegmentation.directing_vision && (() => {
                      const dv = state.scriptSegmentation.directing_vision;
                      return (
                        <div className="space-y-6">
                          <div className="text-center space-y-1 pb-2">
                            <div className="text-[9px] tracking-[0.4em] text-gold/30 uppercase font-mono">Directing Vision — Locked</div>
                            <h2 className="font-serif italic text-2xl text-white/80">Master Blueprint</h2>
                          </div>

                          <div className="grid grid-cols-2 gap-4">
                            {[
                              { label: 'Voice Fingerprint', value: dv.voice_fingerprint },
                              { label: 'Presentation Persona', value: dv.presentation_persona },
                              { label: 'Master Through-Action', value: dv.through_action },
                              { label: 'Character Through-Line', value: dv.character_through_line },
                            ].filter(i => i.value).map(item => (
                              <div key={item.label} className="bg-slate-950/60 border border-white/[0.04] rounded-xl p-5">
                                <div className="text-[8px] tracking-[0.25em] text-slate-700 uppercase font-mono mb-2">{item.label}</div>
                                <p className="text-slate-300 text-sm leading-relaxed italic">"{item.value}"</p>
                              </div>
                            ))}
                          </div>

                          {dv.energy_arc_map && (
                            <div className="bg-black/40 border border-white/[0.04] rounded-xl p-5">
                              <div className="text-[8px] tracking-[0.25em] text-slate-700 uppercase font-mono mb-2">Energy Arc Map</div>
                              <p className="text-slate-400 text-sm leading-relaxed font-mono">{dv.energy_arc_map}</p>
                            </div>
                          )}

                          {state.scriptSegmentation.narrative_arc && (
                            <div className="bg-black/40 border border-white/[0.04] rounded-xl p-5">
                              <div className="text-[8px] tracking-[0.25em] text-slate-700 uppercase font-mono mb-2">Narrative Arc</div>
                              <p className="text-slate-400 text-sm leading-relaxed">{state.scriptSegmentation.narrative_arc}</p>
                            </div>
                          )}

                          {/* Bible Intelligence Summary */}
                          {(() => {
                            const scenes = state.scriptSegmentation!.scenes;
                            const gearCounts = [1,2,3,4].map(g => ({ gear: g, count: scenes.filter(s => s.acting_blueprint?.vocal_gear === g).length })).filter(x => x.count > 0);
                            const avgRetention = scenes.filter(s => s.retention_target_percent).reduce((a,s,_,arr) => a + (s.retention_target_percent||0)/arr.length, 0);
                            const patternInterrupts = scenes.filter(s => s.is_pattern_interrupt).length;
                            return (
                              <div className="bg-black/30 border border-gold/[0.08] rounded-xl p-5 space-y-3">
                                <div className="text-[8px] tracking-[0.3em] text-gold/40 uppercase font-mono">Bible Intelligence — Video Analysis</div>
                                <div className="grid grid-cols-3 gap-3">
                                  <div className="text-center">
                                    <div className="text-lg font-serif text-gold/70">{avgRetention > 0 ? Math.round(avgRetention) + '%' : '—'}</div>
                                    <div className="text-[8px] text-slate-700 font-mono">Avg Retention Target</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-lg font-serif text-gold/70">{patternInterrupts}</div>
                                    <div className="text-[8px] text-slate-700 font-mono">Pattern Interrupts</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="text-lg font-serif text-gold/70">{scenes.length}</div>
                                    <div className="text-[8px] text-slate-700 font-mono">Total Scenes</div>
                                  </div>
                                </div>
                                {gearCounts.length > 0 && (
                                  <div>
                                    <div className="text-[8px] text-slate-700 font-mono mb-2">Vocal Gear Distribution</div>
                                    <div className="flex gap-2 flex-wrap">
                                      {gearCounts.map(({ gear, count }) => (
                                        <span key={gear} className={`text-[8px] font-mono px-2 py-1 rounded border ${
                                          gear === 4 ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                                          gear === 3 ? 'bg-orange-500/10 text-orange-400 border-orange-500/20' :
                                          gear === 2 ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                                          'bg-purple-500/10 text-purple-400 border-purple-500/20'
                                        }`}>
                                          Gear {gear} × {count}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })()}

                          <div className="text-center pt-4 border-t border-white/[0.04]">
                            <p className="text-[9px] text-slate-800 font-mono tracking-widest uppercase">← Select a scene from the sidebar to begin engineering</p>
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                </div>

              ) : (() => {
                const scene = state.scriptSegmentation!.scenes[state.selectedSceneIndex];
                return (
                  <div className="max-w-4xl mx-auto px-8 py-8 space-y-7 pb-24">

                    {/* Scene Header */}
                    <div className="flex items-end justify-between border-b border-white/[0.05] pb-6">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className={`inline-block px-2.5 py-0.5 rounded-full text-[9px] font-bold tracking-widest uppercase border ${roleColor(scene.role)}`}>{scene.role}</span>
                          {scene.bible_scene_type && <span className="text-[8px] text-slate-600 font-mono border border-slate-800 px-2 py-0.5 rounded">{scene.bible_scene_type}</span>}
                          {scene.is_pattern_interrupt && <span className="text-[8px] text-amber-400 font-mono border border-amber-500/25 px-2 py-0.5 rounded bg-amber-500/8">⚡ Pattern Interrupt</span>}
                          {vocalGearBadge(scene.acting_blueprint?.vocal_gear)}
                          {scene.retention_target_percent && <span className="text-[8px] font-mono text-slate-600">Target: {scene.retention_target_percent}% retention</span>}
                          {scene.lip_sync_risk && (() => {
                            const riskMap = { low: 'text-emerald-600 border-emerald-600/20', medium: 'text-amber-500 border-amber-500/20', high: 'text-red-500 border-red-500/20' };
                            return <span className={`text-[8px] font-mono border px-2 py-0.5 rounded ${riskMap[scene.lip_sync_risk!]}`}>Lip Sync {scene.lip_sync_risk?.toUpperCase()}</span>;
                          })()}
                          {!wordCountOk(scene) && <span className="text-[8px] font-mono text-red-500 border border-red-500/20 px-2 py-0.5 rounded bg-red-500/8">⚠ {scene.word_count}w exceeds WPM limit</span>}
                        </div>
                        <h2 className="font-serif italic text-4xl text-white leading-tight">{scene.title}</h2>
                        {scene.gravity_center_word && <p className="text-[10px] text-gold/40 font-mono">Gravity center: "{scene.gravity_center_word}"</p>}
                      </div>
                      <div className="text-right flex flex-col items-end gap-1.5 shrink-0">
                        <p className="font-mono text-gold text-2xl">#{String(scene.scene_number).padStart(2, '0')}</p>
                        <p className={`font-mono text-xs ${scene.duration_seconds > roleMaxSeconds(scene.role) ? 'text-red-400' : scene.duration_seconds <= 5.5 ? 'text-amber-500/70' : 'text-slate-600'}`}>
                          {scene.duration_seconds}s{scene.duration_seconds <= 5.5 ? ' ⚡' : ''}
                        </p>
                        {state.selectedSceneIndex < state.scriptSegmentation!.scenes.length - 1 && (
                          <button onClick={handleMergeNextScene}
                            className="mt-1 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-slate-500 hover:text-slate-300 text-[10px] font-mono rounded-lg border border-white/[0.05] transition-all flex items-center gap-1.5">
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /></svg>
                            Merge Next
                          </button>
                        )}
                      </div>
                    </div>

                    {/* ── IDLE STATE ─────────────────────────────── */}
                    {state.sceneProcessing === 'idle' && (
                      <div className="space-y-6">

                        {/* Script + Blueprint */}
                        <div className="grid grid-cols-5 gap-5">
                          <div className="col-span-3 bg-black/50 border border-white/[0.05] p-7 rounded-2xl relative overflow-hidden">
                            <div className="absolute inset-y-0 left-0 w-px bg-gradient-to-b from-transparent via-gold/25 to-transparent" />
                            <div className="text-[9px] tracking-[0.25em] text-slate-700 uppercase font-mono mb-4">Script Segment</div>
                            {(() => {
                              const emphasisWords = scene.acting_blueprint.emphasis_words || [];
                              let annotated = scene.script_text;
                              // Build a regex of all emphasis words
                              if (emphasisWords.length > 0) {
                                const parts: React.ReactNode[] = [];
                                let remaining = annotated;
                                const pattern = emphasisWords.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
                                const regex = new RegExp(`(${pattern})`, 'gi');
                                let lastIndex = 0;
                                let match: RegExpExecArray | null;
                                regex.lastIndex = 0;
                                const matches: { index: number; word: string }[] = [];
                                while ((match = regex.exec(annotated)) !== null) {
                                  matches.push({ index: match.index, word: match[0] });
                                }
                                if (matches.length === 0) {
                                  return <p className="font-serif italic text-2xl text-white/90 leading-relaxed">"{scene.script_text}"</p>;
                                }
                                const nodes: React.ReactNode[] = ['"'];
                                let prev = 0;
                                matches.forEach(({ index, word }) => {
                                  if (index > prev) nodes.push(annotated.slice(prev, index));
                                  nodes.push(<span key={index} className="text-gold/90 underline decoration-gold/30 underline-offset-4">{word}</span>);
                                  prev = index + word.length;
                                });
                                if (prev < annotated.length) nodes.push(annotated.slice(prev));
                                nodes.push('"');
                                return <p className="font-serif italic text-2xl text-white/90 leading-relaxed">{nodes}</p>;
                              }
                              return <p className="font-serif italic text-2xl text-white/90 leading-relaxed">"{scene.script_text}"</p>;
                            })()}
                            <div className="mt-5 flex flex-wrap gap-2">
                              {[
                                `Energy ${scene.energy_level}/10`,
                                `${scene.acting_blueprint.delivery_pace_wpm} WPM`,
                                scene.emotional_tone
                              ].map(tag => (
                                <span key={tag} className="bg-slate-900/80 text-slate-600 text-[9px] px-2 py-1 rounded font-mono border border-white/[0.04]">{tag}</span>
                              ))}
                            </div>
                          </div>
                          <div className="col-span-2 bg-slate-950/50 border border-white/[0.04] p-5 rounded-2xl space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="text-[9px] tracking-[0.25em] text-slate-700 uppercase font-mono">Acting Blueprint</div>
                              <div className="flex items-center gap-1.5">
                                {vocalGearBadge(scene.acting_blueprint?.vocal_gear)}
                                {retentionBadge(scene.retention_target_percent)}
                                {scene.acting_blueprint?.forward_lean && <span className="text-[7px] text-amber-400 font-mono border border-amber-500/25 px-1 py-0.5 rounded bg-amber-500/8">10° FWD</span>}
                                {scene.acting_blueprint?.pregnant_pause_required && <span className="text-[7px] text-purple-400 font-mono border border-purple-500/25 px-1 py-0.5 rounded bg-purple-500/8">PAUSE</span>}
                              </div>
                            </div>
                            {scene.acting_blueprint?.vocal_gear && (
                              <div className="bg-black/30 rounded-lg px-3 py-2 border border-white/[0.04]">
                                <div className="text-[7px] text-slate-700 uppercase tracking-wider font-mono mb-0.5">
                                  {scene.acting_blueprint.vocal_gear === 4 ? 'Gear 4 — Passionate/Driving' :
                                   scene.acting_blueprint.vocal_gear === 3 ? 'Gear 3 — Engaged/Clear' :
                                   scene.acting_blueprint.vocal_gear === 2 ? 'Gear 2 — Conversational/Warm' :
                                   'Gear 1 — Intimate/Quiet'}
                                </div>
                                {scene.acting_blueprint.vocal_gear_rationale && (
                                  <p className="text-slate-600 text-[9px] leading-relaxed">{scene.acting_blueprint.vocal_gear_rationale}</p>
                                )}
                              </div>
                            )}
                            {scene.gravity_center_word && (
                              <div>
                                <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-0.5">Gravity Center</div>
                                <p className="text-gold/70 text-xs font-mono">"{scene.gravity_center_word}"</p>
                              </div>
                            )}
                            <div>
                              <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-0.5">Intention</div>
                              <p className="text-slate-400 text-xs leading-relaxed">{scene.acting_blueprint.intention}</p>
                            </div>
                            <div>
                              <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-0.5">Subtext</div>
                              <p className="text-slate-500 italic text-xs">"{scene.acting_blueprint.subtext}"</p>
                            </div>
                            {scene.recommended_b_roll && (
                              <div className="bg-emerald-950/20 border border-emerald-900/30 rounded p-2">
                                <div className="text-[8px] text-emerald-600 uppercase tracking-wider font-mono mb-0.5 flex items-center gap-1">
                                  <svg className="w-2.5 h-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                                  Recommended B-Roll
                                </div>
                                <p className="text-emerald-400 text-[10px] leading-relaxed font-mono">{scene.recommended_b_roll}</p>
                              </div>
                            )}
                            <div>
                              <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-1">Emphasis</div>
                              <div className="flex flex-wrap gap-1">
                                {scene.acting_blueprint.emphasis_words.map((w, i) => (
                                  <span key={i} className="bg-gold/8 text-gold/60 text-[8px] px-2 py-0.5 rounded font-mono border border-gold/10">{w}</span>
                                ))}
                              </div>
                            </div>
                            {scene.acting_blueprint.focal_length && (
                              <div>
                                <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-0.5">Lens</div>
                                <p className="text-slate-600 text-[9px]">{scene.acting_blueprint.focal_length}</p>
                              </div>
                            )}
                            {(scene.acting_blueprint as any).speech_onset_phoneme && (
                              <div>
                                <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-0.5">Speech Onset</div>
                                <p className="text-slate-600 text-[9px] font-mono leading-relaxed">{(scene.acting_blueprint as any).speech_onset_phoneme}</p>
                              </div>
                            )}
                            {scene.acting_blueprint.lip_sync_blueprint?.phonemic_anchors && scene.acting_blueprint.lip_sync_blueprint.phonemic_anchors.length > 0 && (
                              <div>
                                <div className="text-[8px] text-slate-700 uppercase tracking-wider font-mono mb-1">Phonemic Anchors</div>
                                <div className="space-y-0.5">
                                  {scene.acting_blueprint.lip_sync_blueprint.phonemic_anchors.slice(0, 3).map((anchor, i) => (
                                    <p key={i} className="text-slate-700 text-[8px] font-mono leading-relaxed">{anchor}</p>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Scene Composition Summary */}
                        <div className="bg-black/30 border border-white/[0.04] rounded-2xl p-5 space-y-3">
                          <div className="text-[9px] tracking-[0.25em] text-slate-700 uppercase font-mono">Scene Composition — Derived from Acting Intelligence</div>
                          <div className="grid grid-cols-2 gap-4 text-[9px] font-mono">
                            <div>
                              <div className="text-slate-700 uppercase tracking-wider mb-1">Opening State</div>
                              <p className="text-slate-500 leading-relaxed">{(scene.acting_blueprint as any).moment_before || scene.acting_blueprint.physical_signature || '—'}</p>
                            </div>
                            <div>
                              <div className="text-slate-700 uppercase tracking-wider mb-1">Exit State</div>
                              <p className="text-slate-500 leading-relaxed">{scene.continuity.exits_to || '—'}</p>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 pt-1 border-t border-white/[0.03]">
                            <span className="w-1 h-1 rounded-full bg-gold/40" />
                            <span className="text-[8px] text-slate-700 font-mono">Character identity locked from uploaded photos · Environment derived from photo background</span>
                          </div>
                        </div>

                        {/* Engineer Button */}
                        <button
                          disabled={state.sceneProcessing === 'engineering'}
                          onClick={handleEngineerScene}
                          className="w-full py-5 bg-white text-black text-lg font-serif italic rounded-xl hover:bg-slate-50 disabled:opacity-25 disabled:cursor-not-allowed transition-all duration-300 shadow-[0_0_30px_rgba(255,255,255,0.04)] hover:shadow-[0_0_50px_rgba(255,255,255,0.1)] tracking-wide"
                        >
                          Engineer Elite VEO 3.1 Prompt →
                        </button>
                      </div>
                    )}

                    {/* ── ENGINEERING STATE ──────────────────────── */}
                    {state.sceneProcessing === 'engineering' && (
                      <div className="py-28 flex flex-col items-center space-y-8">
                        <div className="relative w-20 h-20">
                          <div className="absolute inset-0 border-[3px] border-t-gold border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" />
                          <div className="absolute inset-2.5 border-2 border-t-gold/20 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin" style={{ animationDuration: '2.2s', animationDirection: 'reverse' }} />
                        </div>
                        <div className="text-center space-y-3">
                          <h3 className="font-serif italic text-3xl text-white">Engineering Scene #{String(scene.scene_number).padStart(2, '0')}</h3>
                          <p className="text-gold/35 font-mono text-[10px] tracking-[0.25em] uppercase">{state.sceneProcessingStatus}</p>
                          <div className="flex items-center justify-center gap-6 mt-2 text-[9px] font-mono">
                            {['Pass 1 — Annotation', 'Pass 2 — Engineering', 'Pass 3 — Distilling'].map((label, i) => {
                              const status = state.sceneProcessingStatus;
                              const isActive = (i === 0 && status.includes('Pass 1')) || (i === 1 && status.includes('Pass 2')) || (i === 2 && status.includes('Pass 3'));
                              const isDone   = (i === 0 && (status.includes('Pass 2') || status.includes('Pass 3'))) || (i === 1 && status.includes('Pass 3'));
                              return (
                                <span key={i} className={`flex items-center gap-1.5 ${isDone ? 'text-emerald-600' : isActive ? 'text-gold/70' : 'text-slate-800'}`}>
                                  <span>{isDone ? '✓' : isActive ? '●' : '○'}</span>{label}
                                </span>
                              );
                            })}
                          </div>
                          <p className="text-slate-800 text-[10px] font-mono mt-1">3 high-thinking passes — 90–180 seconds</p>
                        </div>
                      </div>
                    )}

                    {/* ── COMPLETE STATE ─────────────────────────── */}
                    {state.sceneProcessing === 'complete' && state.currentPrompt && (
                      <div className="space-y-5">

                        {/* Status bar */}
                        <div className="flex items-center justify-between bg-emerald-950/20 border border-emerald-800/20 p-4 rounded-xl">
                          <div className="flex items-center gap-2.5 text-emerald-500">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_6px_rgba(52,211,153,0.5)]" />
                            <span className="font-mono text-[10px] tracking-widest uppercase">VEO 3.1 Prompt Ready — 3-Pass Elite</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-[9px] text-slate-700 font-mono">{state.currentPrompt.split(/\s+/).filter(Boolean).length} words</span>
                            <button onClick={copyPrompt}
                              className={`px-5 py-2 rounded-lg font-mono text-[10px] tracking-widest uppercase font-bold transition-all duration-300 ${copied ? 'bg-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.35)]' : 'bg-gold hover:bg-yellow-500 text-black shadow-[0_0_15px_rgba(202,138,4,0.2)]'}`}>
                              {copied ? 'Copied ✓' : 'Copy Prompt'}
                            </button>
                          </div>
                        </div>

                        {/* Star Rating + Energy Calibration */}
                        <div className="flex items-center gap-4 px-4 py-3 bg-black/30 border border-white/[0.04] rounded-xl">
                          <span className="text-[9px] font-mono text-slate-700 tracking-widest uppercase shrink-0">Rate VEO Output</span>
                          <div className="flex gap-0.5">
                            {[1,2,3,4,5].map(star => {
                              const completed = state.completedScenes.find(c => c.scene_number === scene.scene_number);
                              const rating = completed?.rating || 0;
                              return (
                                <button key={star} onClick={() => handleRateScene(scene.scene_number, star)}
                                  className={`text-xl transition-all duration-150 hover:scale-125 ${star <= rating ? 'text-gold' : 'text-slate-800 hover:text-gold/40'}`}
                                  title={star === 1 ? 'Poor' : star === 2 ? 'Fair' : star === 3 ? 'Good' : star === 4 ? 'Great' : 'Perfect'}>
                                  ★
                                </button>
                              );
                            })}
                          </div>
                          {state.learnedPreferences && (state.learnedPreferences.gear || state.learnedPreferences.focal) && (
                            <div className="flex items-center gap-2 ml-2">
                              <span className="w-1 h-1 rounded-full bg-emerald-500 shadow-[0_0_4px_rgba(16,185,129,0.6)]" />
                              <span className="text-[8px] font-mono text-emerald-600">
                                Calibrated: {[
                                  state.learnedPreferences.gear ? `G${state.learnedPreferences.gear}` : null,
                                  state.learnedPreferences.focal,
                                  state.learnedPreferences.energy ? `E${state.learnedPreferences.energy}` : null,
                                ].filter(Boolean).join(' · ')}
                              </span>
                            </div>
                          )}
                          <span className="text-[8px] text-slate-800 font-mono ml-auto">Rate after seeing VEO output to calibrate future scenes</span>
                        </div>

                        {/* Prompt display */}
                        <div className="bg-black/70 border border-white/[0.05] rounded-2xl p-8 max-h-[560px] overflow-y-auto custom-scrollbar shadow-2xl relative">
                          <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-gold/15 to-transparent" />
                          <pre className="font-sans text-[13px] text-slate-200 leading-[1.95] whitespace-pre-wrap selection:bg-gold/15 font-light">{state.currentPrompt}</pre>
                        </div>

                        {/* ── REFINE LOOP ─────────────────────────────── */}
                        <div className="border border-white/[0.05] rounded-xl overflow-hidden">
                          {/* Refine header */}
                          <button
                            onClick={() => setState(s => ({ ...s, showRefineInput: !s.showRefineInput }))}
                            className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-white/[0.02] transition-all group"
                          >
                            <div className="flex items-center gap-2.5">
                              <svg className="w-3.5 h-3.5 text-gold/50 group-hover:text-gold/80 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                              </svg>
                              <span className="text-[10px] font-mono text-slate-600 group-hover:text-slate-400 tracking-widest uppercase transition-colors">
                                Refine After Seeing VEO Output
                              </span>
                            </div>
                            <span className="text-slate-700 text-xs">{state.showRefineInput ? '▲' : '▼'}</span>
                          </button>

                          {/* Refine form */}
                          {state.showRefineInput && (
                            <div className="px-5 pb-5 space-y-3 border-t border-white/[0.04] pt-4">
                              <p className="text-[10px] text-slate-600 font-mono leading-relaxed">
                                Run VEO with the prompt above. Then describe what needs fixing — lip sync on a specific word, camera movement, face drift, energy, accent. One cycle typically closes 80% of the gap.
                              </p>
                              <textarea
                                value={state.refinementFeedback}
                                onChange={e => setState(s => ({ ...s, refinementFeedback: e.target.value }))}
                                placeholder={'e.g. "Lip sync was off on \'capital\' — lips didn\'t fully close. Also the camera moved too much — should be locked."'}
                                rows={3}
                                className="w-full bg-black/60 border border-white/[0.07] rounded-lg px-4 py-3 text-slate-300 placeholder:text-slate-700 focus:border-gold/20 focus:outline-none font-mono text-xs leading-relaxed resize-none transition-colors"
                              />
                              <div className="flex gap-3">
                                <button
                                  onClick={() => setState(s => ({ ...s, showRefineInput: false, refinementFeedback: '' }))}
                                  className="px-4 py-2 text-slate-600 hover:text-slate-400 font-mono text-[10px] tracking-widest uppercase transition-colors"
                                >
                                  Cancel
                                </button>
                                <button
                                  disabled={!state.refinementFeedback.trim() || state.isRefining}
                                  onClick={handleRefineScene}
                                  className="flex-1 py-2.5 bg-gold/90 hover:bg-gold text-black font-mono text-[10px] tracking-widest uppercase font-bold rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-[0_0_15px_rgba(202,138,4,0.15)]"
                                >
                                  {state.isRefining ? (
                                    <span className="flex items-center justify-center gap-2">
                                      <span className="w-2.5 h-2.5 border border-black/40 border-t-black rounded-full animate-spin" />
                                      Refining with Gemini 3.1 Pro...
                                    </span>
                                  ) : 'Submit Refinement →'}
                                </button>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Action buttons */}
                        <div className="flex gap-4">
                          <button onClick={() => setState(s => ({ ...s, sceneProcessing: 'idle', currentPrompt: null, showRefineInput: false, refinementFeedback: '' }))}
                            className="flex-1 py-3.5 border border-white/[0.06] text-slate-600 hover:text-slate-300 hover:bg-white/[0.02] rounded-xl transition-all font-mono text-[10px] tracking-widest uppercase">
                            Re-Engineer
                          </button>
                          <button
                            disabled={(state.selectedSceneIndex || 0) >= state.scriptSegmentation!.scenes.length - 1}
                            onClick={() => { const n = (state.selectedSceneIndex || 0) + 1; if (n < state.scriptSegmentation!.scenes.length) selectScene(n); }}
                            className="flex-1 py-3.5 bg-white text-black font-serif italic text-lg rounded-xl hover:bg-slate-50 transition-all shadow-[0_0_20px_rgba(255,255,255,0.05)] hover:shadow-[0_0_30px_rgba(255,255,255,0.1)] disabled:opacity-25 disabled:cursor-not-allowed">
                            Next Scene →
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
