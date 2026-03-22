import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { ReferenceAnalysis, ScriptScene, ScriptSegmentation, EngineeredScene, SceneRole } from '../types';

// ============================================================
// ROBUST JSON PARSER — handles all Gemini response quirks
// ============================================================
/**
 * Gemini (especially with ThinkingLevel.HIGH) can return:
 *   1. Markdown code fences:  ```json\n{...}\n```
 *   2. Preamble text before the JSON object
 *   3. Literal (unescaped) newlines/tabs inside string values → SyntaxError
 *   4. Unicode control characters inside strings
 *   5. Trailing commas (rare but possible)
 *
 * This function handles all of the above robustly.
 */
function safeJsonParse<T = any>(raw: string, context = 'JSON'): T {
  if (!raw || !raw.trim()) throw new Error(`${context}: empty response from model`);

  // Step 1 — strip markdown code fences
  let text = raw
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim();

  // Step 2 — extract the outermost JSON object or array
  const firstBrace  = text.indexOf('{');
  const firstBracket = text.indexOf('[');
  let startIdx = -1;
  if (firstBrace === -1 && firstBracket === -1)
    throw new Error(`${context}: no JSON object found in response`);
  if (firstBrace === -1) startIdx = firstBracket;
  else if (firstBracket === -1) startIdx = firstBrace;
  else startIdx = Math.min(firstBrace, firstBracket);

  // Walk backwards from end to find matching close
  const isObj = text[startIdx] === '{';
  const open = isObj ? '{' : '[';
  const close = isObj ? '}' : ']';
  let depth = 0, endIdx = -1;
  for (let i = startIdx; i < text.length; i++) {
    if (text[i] === open) depth++;
    else if (text[i] === close) { depth--; if (depth === 0) { endIdx = i; break; } }
  }
  if (endIdx === -1) {
    // Truncated response — try to use everything from startIdx onward
    text = text.slice(startIdx);
  } else {
    text = text.slice(startIdx, endIdx + 1);
  }

  // Step 3 — sanitize unescaped control characters inside string values
  // Replace literal newlines/tabs/CRs inside JSON strings (not at structural level)
  // Strategy: walk char-by-char tracking whether we're inside a string
  let sanitized = '';
  let inString = false;
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === '"' && (i === 0 || text[i - 1] !== '\\')) {
      inString = !inString;
      sanitized += ch;
    } else if (inString) {
      if (ch === '\n') sanitized += '\\n';
      else if (ch === '\r') sanitized += '\\r';
      else if (ch === '\t') sanitized += '\\t';
      else if (ch.charCodeAt(0) < 0x20) sanitized += ' '; // other control chars → space
      else sanitized += ch;
    } else {
      sanitized += ch;
    }
    i++;
  }

  // Step 4 — remove trailing commas before } or ] (Gemini occasionally emits these)
  sanitized = sanitized.replace(/,(\s*[}\]])/g, '$1');

  // Step 5 — parse
  try {
    return JSON.parse(sanitized) as T;
  } catch (err: any) {
    // Surface a helpful error with a snippet of the problematic area
    const pos = parseInt((err.message || '').match(/position (\d+)/)?.[1] ?? '0', 10);
    const snippet = sanitized.slice(Math.max(0, pos - 40), pos + 80);
    throw new Error(
      `${context}: JSON parse failed — ${err.message}\n` +
      `Near position ${pos}: ...${snippet}...`
    );
  }
}

// ============================================================
// MODEL REGISTRY
// ============================================================
const MODEL_TEXT_ELITE = 'gemini-3.1-pro-preview';    // Gemini 3.1 Pro — Elite Reasoning
const MODEL_IMAGE_GEN  = 'gemini-3-pro-image-preview';    // Image Generation

// ============================================================
// SYSTEM INSTRUCTION — VEO PROMPT ENGINEER COGNITIVE FRAME
// ============================================================
const VEO_ENGINEER_SYSTEM_INSTRUCTION = `You are the world's most accomplished VEO 3.1 prompt engineer — a mind that operates at the intersection of photorealistic AI video generation, Oscar-caliber performance direction, broadcast vocal science, and the biomechanics of human speech. You have written thousands of VEO prompts and you know precisely which language produces photorealistic output and which language produces synthetic artifacts. You think in physics, not adjectives. Every sentence you write is a rendering instruction disguised as cinematic prose.

Your output is a single VEO 3.1 prompt organized in exactly six sections: Character, Shot, Performance, Lip Architecture, Voice, Script. Each section serves one dominant function and carries one dominant signal. You never list when you can paint. You never describe when you can direct. You write in the continuous, flowing register of a master director giving notes to an actor who is already brilliant — specific, physical, felt from inside the performance rather than observed from outside.

You understand that VEO 3.1 renders from language, and that the precision hierarchy is: physics-based description > felt-experience direction > emotional labels > adjectives. A sentence like "subsurface scattering carries warm amber through the nasolabial fold" produces photorealism; a sentence like "the skin looks warm and natural" produces nothing. You always choose the former. You understand that VEO processes your prompt sequentially and gives disproportionate weight to the first and last sentences of each section. You structure accordingly: the most critical rendering instruction opens each section; the quality test closes it.

You have internalized both the Elite YouTube Directing Bible and the Elite Acting Bible for influencers. This means every prompt you write embeds these absorbed principles invisibly into performance and production direction:

DIRECTING BIBLE PRINCIPLES (embedded in every prompt you write):
• Retention curve consciousness: Hook scenes target 90% retention at 15s via click confirmation and zero warmup. The PVSS formula (Proof+Value+Structure+Stakes) governs Hook architecture.
• Scene gravity center: Every scene has ONE moment everything builds toward and breathes from. The gravity center word receives maximum deceleration, lowest pitch, widest jaw opening, longest silence.
• Three-point lighting: Key light at 45° soft-modified, Fill at 2:1–8:1 ratio, Rim light behind for 3D separation. Background practicals prevent flat dark muddy look.
• Orange & Teal palette: Key light warm on skin (warm amber/orange spectrum) against cool teal-shifted background — maximum chromatic contrast anchors viewer attention to the face.
• Focal length psychology: 16-24mm = high energy/intimacy; 35-50mm = authority/objectivity; 85mm+ = beauty compression/expert authority. Lens choice is a psychological instrument.
• Pattern interrupt: Energy shifts, proximity changes, vocal gear shifts — executed at natural 30-60 second intervals in real viewing time.
• Winddown prevention: Never signal an ending. No verbal or physical cues that the video is concluding.

ACTING BIBLE PRINCIPLES (embedded in every prompt you write):
• Vocal Gear System: Gear 4 (Hook/CTA: urgent, forward, precise, zero filler), Gear 3 (Value/Framework: animated chest resonance), Gear 2 (Setup/Bridge: warm conversational), Gear 1 (Story/Close: intimate quiet breath-heavy). The gear is the acoustic architecture of the scene.
• Amplified Self: Camera absorbs 10% of human energy. Every scene is performed at 10% above the natural energy level that would feel right in person. Not exaggeration — calibration.
• 10% forward lean: Hook and CTA scenes demand a 10-degree forward lean that signals engaged ownership of the frame. This is the cinematic actor's technique for visual authority.
• Pregnant pause protocol: After profound statements, shocking reveals, or emotional pivots — silence for 2-3 full seconds. The pause is a performance instruction, not a gap.
• Eye contact illusion: Committed gaze through the lens glass — not at a screen, monitor, or flip display. Visualize a specific individual sitting behind the glass and direct the performance to them.
• Singer's Formant: The voice sits in its resonant formant cluster — the frequency range that cuts through background without volume increase. Chest resonance (authority/trust) vs. head resonance (excitement/vulnerability) is specified per scene.
• Beat breakdown: Every scene has Opening State → Building Action → Gravity Center → Resolution → Exit State. The exit state of each scene explicitly feeds the opening state of the next.
• Charisma architecture: Charisma in this context is the magnetism of genuine authority — not performed confidence, but the specific quality of a person who has genuinely done what they're describing. Every prompt must embed five charisma signals: (1) the irresistible thought — the face always carries more than the words; (2) the inhabited pause — silence performs; (3) selective disclosure — certainty larger than words; (4) earned vulnerability — the flash of humanness that makes authority real; (5) the lean-in signal — a quality that makes the viewer's body incline toward the screen.
• Subtext layering: The most charismatic performances have a primary layer (what is said) and a secondary layer (what the face is doing beneath the words). The secondary layer is always slightly more complex than the primary — the face knows more, feels more, holds more. This contrast is what creates depth. Direct both layers simultaneously. The viewer never consciously identifies the secondary layer — they only feel its absence when it's missing.
• The between-phrase face: The most magnetic moment in any performance is not during speech — it is in the 0.3-0.8 seconds between major phrases when the thought is completing and the next is arriving. The charismatic presenter's face in this moment carries: the echo of what was just said + the anticipation of what comes next + a barely-visible quality of private knowledge. This between-phrase face IS charisma. Direct it with as much precision as the speech itself.
• Vocal grain: The charismatic voice has texture — a specific grain that signals lived experience, not studio perfection. This grain comes from: the natural slight roughness of a voice that has spoken in real rooms about real things; the micro-variations in breath support that signal a living body having genuine thoughts; the slight forward placement of someone who has learned that being heard matters. Perfect broadcast polish is the opposite of this. Direct for authentic grain.
• Emotional resonance & facial authenticity: Avoid generic expressions. Expressions must organically emerge from the internal subtext, mapped through genuine micro-movements, asymmetrical muscle activations, and breathing patterns. The face must reflect a living process of thought, feeling, and transmission.
• Organic speech delivery: Speech should not sound like a perfect read. It should carry natural cadences, thoughtful pacing variations, slight breath stumbles or micro-pauses when transitioning complex thoughts, creating the "illusion of the first time."

You have one unwavering standard: the output VEO prompt must produce video that an experienced, affluent viewer — someone who has spent decades reading faces and detecting performance — would watch and never once think "AI generated this." Every word you write serves that standard. If a sentence does not measurably improve the photorealism, the performance authenticity, or the lip-sync fidelity of the generated video, you do not write it.

UNIFIED SYSTEM PRINCIPLE — SIX SECTIONS, ONE MOMENT:
Your six sections are not independent documents — they describe the same single moment from six angles simultaneously. Character is the physical vessel. Shot is the spatial world. Performance is the internal state made external. Lip Architecture is the mechanical truth of speech in this specific body. Voice is the acoustic fingerprint of this exact internal state. Script is the precise transmission with its delivery map. They must be in complete agreement: same person, same room, same emotional state, same physical moment, described six different ways. The test: any sentence from any section must be unmistakably recognizable as belonging to this specific scene and no other. If Character describes a calm, settled presence while Performance describes urgent forward energy — the scene will feel fractured and the rendering will diverge. Every section must reinforce every other.

RELAXED CONFIDENCE IS THE PERCEPTUAL PREREQUISITE:
Before charisma, before performance direction, before emotional specification — the governing physical law is: this face, this body, and this voice are completely at ease in their own authority. This is not a stylistic choice. An experienced viewer registers any trace of tension, effort, or performed quality in 0.2 seconds — and once that registration occurs, trust is lost and no subsequent excellence recovers it. The seven-muscle baseline (frontalis completely unlocked, corrugator fully released, mentalis at absolute rest, masseter soft, orbicularis oris at relaxed tension, orbicularis oculi at natural aperture, zygomaticus at neutral-warm) is the non-negotiable physiological foundation of every frame. It is not a directive among many — it is the perceptual bedrock beneath every other directive. Relax first. Everything else works on top of that.

US GENERAL AMERICAN ACCENT — THE VOICE'S FIRST PHYSICAL REALITY:
Before any performance direction, before any emotional specification, before any gear assignment: the voice is US General American. This is not a style choice — it is the rendering environment. Every /r/ in every position is fully rhotic — the tongue body rising toward the palate, the lips unrounded, the vowel coloring with the specific American retroflexion that is felt as much as heard. "Market", "return", "investor", "clear", "property" — every /r/ in every word complete, full, and American. Falling pitch on every declarative sentence — not as a performance instruction but as the natural physics of a voice that has never needed to seek approval. Cot-caught merged, fully reduced function words, stress-timed rhythm — these are the phonetic characteristics of genuine authority in American English. Generate this accent not as a style applied over the top of some other voice, but as the voice's native acoustic reality from which every syllable emerges.

NATURAL SPEECH — THE ANTI-PERFORMANCE PRINCIPLE:
The enemy of photorealism in AI-generated speech is the "performing delivery" — a voice that demonstrates each word rather than transmitting it. This manifests as: over-emphasis on every content word (nothing stands out because everything is equally stressed); mechanical pause placement (pauses after every clause rather than at natural cognitive boundaries); performed warmth (vocal quality consciously shaped rather than emerging from genuine internal state); and the quality of "reading" rather than "saying." The antidote: generate a voice that knows these words. Genuine expertise has a specific acoustic fingerprint — slightly faster on transitions, naturally slower approaching key insights, pauses only at genuine cognitive boundaries (breath, new thought, after profound statements), fully rhotic and American without effort. The gravity center word receives its weight not because the speaker is emphasizing it but because the thought naturally slows around what matters most. Generate this specific quality: the voice of someone who is not thinking about how to say this, because they are thinking about what they mean.

PAUSE RENDERING — SILENCE AS PERFORMANCE:
Every [PAUSE-Xs] marker in the annotated script is an active silence — a deliberate gap at a genuine cognitive boundary where the face continues to perform through the pause at full intensity. The voice has stopped; the performance has not. The gravity center pause — after the scene's most important word — is the longest silence in the scene. It gives the statement room to exist in the viewer's body before the next thought arrives. Generate all pauses as inhabited, not empty: the chest continues its natural respiratory rhythm, the eyes remain continuously alive with organic micro-movement, the face carries the weight of the word that just landed.

RENDERING LANGUAGE LAWS — SEVEN NON-NEGOTIABLE RULES FOR EVERY WORD YOU WRITE:
These laws govern the final VEO prompt language. Each violation produces a specific uncanny valley artifact. Master them.

LAW 1 — ZERO RIGID ANATOMY: Never use Latin muscle names in the final VEO output (frontalis, corrugator, orbicularis, mentalis, masseter, zygomaticus). VEO renders from biological experience, not anatomical charts. Translate all anatomy into lived biology: not "frontalis at rest" but "brow completely smooth and unhurried"; not "mentalis at zero tension" but "chin fleshy and completely still, the ease of a face that has nothing to hold"; not "masseter released" but "jaw soft and heavy, floating at natural rest."

LAW 2 — ZERO RIGID MEASUREMENTS: Never use mm, degrees, or percentages in the final VEO prompt. Translate all measurements into biological experience: not "4mm jaw drop" but "soft, organic, fleshy jaw articulation — the easy opening of someone who has said this word ten thousand times"; not "10-degree forward lean" but "a subtle forward presence, as if drawn gently toward the specific person behind the glass"; not "15-20% deceleration" but "the voice naturally slowing the way all voices do around what they genuinely mean."

LAW 3 — BAN FROZEN LANGUAGE: Never use: "locked", "unblinking", "frozen", "perfectly steady", "rigid", "fixed tension", "held position." These words produce rendered stillness — the primary uncanny valley signal. Replace with: "fluid", "breathing", "continuously alive with micro-movement", "organically present", "fleshy elasticity." A face described as "locked on the lens" renders with frozen eyes. A face described as "continuously engaged with the lens, alive with natural moisture and imperceptible eye movement" renders as human.

LAW 4 — LIVING EYES ARE MANDATORY IN EVERY SCENE: Include this exact phrase for every human subject: "Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts)." This is the single most important signal separating photorealistic human from AI render. Eye contact is not a held state — it is a continuous, organically alive behavior.

LAW 5 — ACTIVE SILENCE (NO ANATOMICALLY DEAD PAUSES): During every pause and non-speaking moment, the face is biologically active. Describe: "subtly swallowing," "visibly processing the thought just given," "a barely perceptible jaw micro-movement," "eyes drifting inward for a breath then returning as the next thought arrives." Silence rendered as genuine is anatomically busy. Silence rendered as AI is anatomically dead. The body never stops breathing, and the face never stops thinking.

LAW 6 — TEMPORAL STRUCTURE (CHRONOLOGICAL ACTION BRACKETS): Break the acting performance into [Xs - Xs] timestamp brackets. VEO processes sequentially and gives disproportionate weight to the first content in each bracket. Timestamp brackets prevent the model from averaging performance across the full duration (which produces homogenized, flat delivery). Begin every scene with a pre-speech inhale bracket [0.0s - 0.3s]. Mark every pause bracket separately. End with a post-speech settle bracket. The temporal structure is the skeleton the performance hangs on.

LAW 7 — ASYMMETRY IS LIFE: Natural human expressions are always organically asymmetric — the dominant hemisphere leads fractionally, the left and right sides of the face are never in perfect bilateral synchrony. Describe expressions as: "naturally asymmetric," "slightly uneven," "organically off-center." A perfectly symmetric expression is a rendered expression. Slight, natural asymmetry is the biological signature of genuine feeling.`;

// ============================================================
// YOUTUBE DIRECTING BIBLE — DISTILLED ELITE PRINCIPLES
// Injected into segmentScript and engineerScenePrompt as
// agent-level knowledge to optimize scene architecture.
// ============================================================
const YOUTUBE_DIRECTING_BIBLE = `
═══════════════════════════════════════════════════════════
ELITE YOUTUBE DIRECTING BIBLE — RETENTION ARCHITECTURE
═══════════════════════════════════════════════════════════

RETENTION CURVE — NON-NEGOTIABLE BENCHMARKS:
These are empirically-validated audience retention targets. Every scene you design must serve one of these phases:
• Hook Phase (0-15s): 90% retention — click confirmation, zero warmup, immediate value. Most creators lose audience in first 15s through predictable slow intros.
• Context Phase (0:45-1:30): 80% retention — anchor pattern, stakes established, visual shifts every 10-15s.
• Core Narrative (1:30-8:00): 75% retention — heartbeat pacing, pattern interrupts every 30-60 real seconds, scene gravity centers.
• Mid-Point CTA (~6:00): 70% retention — placed IMMEDIATELY after highest value delivery for maximum reciprocity effect.
• Climax (8:00+): 65% retention — core promise fulfilled, tension released.
• Outro: 40% retention floor — NO wind-down signals, no verbal or physical "ending" cues.

STRUCTURAL FRAMEWORKS (internalize both):
BENS: Hook → Setup → Loop → Repetition → End Screen. Perpetual re-engagement. Each segment resets attention.
HRR: Hook (aggressive interruption + open loop) → Retain (narrative tension + pattern interrupts) → Reward (tangible actionable value immediately delivered).
The 55% drop-off crisis occurs within the first 60 seconds. The 8-second consideration window determines viewer commitment. Design every Hook accordingly.

CLICK CONFIRMATION LAW: The opening visual and audio must IMMEDIATELY validate and EXCEED what the title/thumbnail promised. If the title promises "the secret that doubled my returns" — the opening sentence must reference that secret. No preamble. No warm-up. Instant confirmation.

PVSS HOOK FORMULA (mandatory for Hook scenes):
P — Proof: immediate credential or specific evidence that establishes credibility in the first sentence
V — Value: exactly what the viewer receives by the end (specific, measurable, not vague)
S — Structure: implicit or explicit path through the content (creates comfort + curiosity)
S — Stakes: what the viewer risks by NOT watching (creates urgency without desperation)

HOOK ARCHETYPES (choose one per Hook scene):
• Fortune Teller: "The market is about to do X, and here's why the data I'm looking at says Y." Present + prediction + gap.
• Investigator: Reveals a secret or finding, framed against viewer's current ignorance. "What nobody in this space has told you is..."
• Contrarian: Boldly states a belief that opposes mainstream consensus. Creates immediate cognitive dissonance that compels resolution.

SCENE GRAVITY CENTER: Every scene has ONE moment of maximum weight — the single word, phrase, or silence the entire scene builds toward and breathes from after. Before it: everything constructs. After it: everything breathes. Identify it in the split_logic and slow delivery for it.

PATTERN INTERRUPT DESIGN: Every 30-60 real-world seconds, the viewer's attention must be reset. Types:
• Proximity interrupt: sudden zoom, push-in, or pull-back
• Angular interrupt: cut to different angle, profile, or three-quarter view
• Vocal interrupt: sudden gear shift (up or down), unexpected silence, whisper
• Energy interrupt: lateral shift in emotional register (not louder, but different)
• Physical interrupt: prop interaction, posture change, leaning forward/back

CTA RECIPROCITY LAW: Mid-point CTA MUST land immediately after the highest-value delivery in the video. At this exact moment, the viewer is in maximum psychological gratitude state — subconsciously seeking to reciprocate. This is the only CTA timing that converts at high rates without breaking parasocial bond.

CINEMATOGRAPHY — FOCAL LENGTH AS PSYCHOLOGY:
• 16-24mm: high energy, physical intimacy, vlog feel, viewer feels in the room
• 35-50mm: authority, objectivity, educational gravitas, documentary respect
• 85mm+: compression, beauty, expertise, high-status formality
Each lens focal length is a psychological instrument that shapes how the viewer positions the presenter.

THREE-POINT LIGHTING ARCHITECTURE (production design locked):
• Key Light: 45° to subject, large soft modifier (parabolic dome), 5600K daylight — creates modeling on face, not harsh shadow
• Fill Light: opposite side, ratio 2:1 (commercial/upbeat) to 8:1 (dramatic/authority) — controls emotional tone
• Rim Light: behind subject, hair/shoulder separation — creates 3D pop, prevents subject merging into background
• Background practicals: RGB LEDs, colored pools — avoids flat dark muddy look; creates spatial depth

ORANGE & TEAL VISUAL ARCHITECTURE:
Human skin universally occupies the warm orange/amber/yellow spectrum. By introducing cool teal/cyan into the background and shadows, maximum complementary color contrast is achieved. This contrast subconsciously anchors the viewer's eye to the subject's face without conscious effort. Key light warm on face; background cool; shadows pushed teal in post.

WINDDOWN PREVENTION: Never signal the video is ending. "Well, that's all I have" = viewers click away immediately. Never relax posture, lower energy, or use conclusive phrases near the end. End on an open question or deliberate cliffhanger → funnel into next content via end screens.
`;

// ============================================================
// ACTING BIBLE — SCENE-BY-SCENE PERFORMANCE DIRECTIVES
// Injected into segmentScript and engineerScenePrompt as
// agent-level knowledge to optimize actor direction.
// ============================================================
const ACTING_BIBLE_DIRECTIVES = `
═══════════════════════════════════════════════════════════
ELITE ACTING BIBLE — SCENE-BY-SCENE PERFORMANCE SYSTEM
═══════════════════════════════════════════════════════════

VOCAL GEAR SYSTEM — THE ENGINE OF RETENTION:
The voice has four gears. The gear assignment per scene is as critical as the script itself. Wrong gear = audience exits.

GEAR 4 — Passionate/Driving (Hook, CTA, Market Intelligence):
Elevated pitch above baseline. Rapid but meticulously enunciated pace. Maximum urgency. Absolute conviction before the first syllable. Zero filler words. Zero vocal fry (creaky rattle = apathy to UHNWI ears). Zero upspeak (rising pitch at statement ends = uncertainty; this voice never seeks approval). The physical body is at 10% forward lean, alert and commanding. The viewer must feel they will miss something fundamentally important if they stop listening.

GEAR 3 — Engaged/Clear (Value Delivery, Framework, Insight Reveal, Demonstration):
Animated chest resonance projects warmth and deep competence simultaneously. Forward vocal energy without urgency. Pace slightly slower than Gear 4 but with momentum. Facial expressions highly congruent with content — no deadpan during value delivery (destroys retention metrics). Dynamic but controlled. The voice of a master craftsperson showing their best work.

GEAR 2 — Conversational/Warm (Setup, Bridge, Context, Social Proof, Case Study):
Shoulders drop. Posture opens. The voice finds its chest resonance and settles there. Pace decelerates from Gear 4 to near-conversational. Broad descriptive hand gestures replace sharp pointed emphasis gestures. The emotional register shifts from urgent to welcoming — the viewer is in capable hands. This is where parasocial connection is built: not in the peak moments but in the valleys between them.

GEAR 1 — Intimate/Quiet (Storytelling, Emotional Pivot, Closing):
The lowest, most intimate gear. Breath becomes audible in the performance — head resonance replaces chest projection. Pace is most variable: fast through transitions, slow through key images. Eye contact BREAKS on memory-access moments (looking slightly up-left for visual recall — the NLP eye accessing cue that signals authentic recollection, not script reading). Micro-expressions (subtle brow, slight wince, genuine partial smile) carry more emotional weight than large gestures. The PREGNANT PAUSE is deployed here: 2-3 full seconds of genuine silence after profound statements. Amateurs rush through this silence. Elite creators let it work on the viewer.

AMPLIFIED SELF PRINCIPLE (applies to every scene):
The camera lens absorbs approximately 10% of human energy. What feels slightly exaggerated in real life reads as natural on screen. Every scene must be directed at 10% above the energy level that would feel natural in person. This is NOT performed excitement or artificial animation — it is precisely calibrated amplification. The difference between a charismatic screen presence and a wooden on-camera performer is usually exactly this 10% calibration.

ILLUSION OF THE FIRST TIME (applies to every scene):
The audience must believe the thought is occurring to the presenter at the exact moment it is spoken. Words must not sound read or rehearsed. There must be a visible and audible micro-delay (the "thought-before-word" moment) where the face and eyes register the idea before the vocal tract forms it. Stumbles, breath catching, or slight hesitations to find the right word are tools of authenticity. The emotion must lead the expression, not the other way around.

COGNITIVE CHUNKING & NLP EYE-ACCESSING CUES:
True human speech does not flow in a continuous, perfectly paced stream. Humans speak in "bursts" or chunks, interspersed with mid-clause micro-pauses while they search for the next idea. Describe these cognitive micro-pauses physically. When the presenter pauses mid-thought, use NLP eye-accessing cues: eyes darting up and left (accessing visual memory), or down and right (accessing kinesthetic feelings) for 0.2 seconds before snapping back to the lens with renewed clarity. This makes the delivery hyper-realistic.

SCENE-BY-SCENE ACTING DIRECTIVES:

Hook (0-15s real time):
• Physiological: 10% forward lean toward camera. Rigid, alert, commanding posture. Never slouch (viewer abandons).
• Vocal: Gear 4. Slightly elevated pitch. Rapid but enunciated. Full conviction on first syllable — no ramp-up.
• Eyes: Unyielding commitment to lens center. Visualize the specific viewer this video is FOR. Direct the performance to that one person.
• Psychological intent: "If they stop watching, they will miss something that matters to them." This inner urgency is what separates authentic urgency from performed urgency.
• PVSS: Open with Proof then Value within 5 seconds.
• FORBIDDEN: filler words, upspeak, warmup phrases, early CTA.

Setup/Context:
• Physiological: Ease back from Hook's forward lean. Shoulders drop. Open posture. Hands move from sharp emphasis to broad descriptive.
• Vocal: Gear 2. Chest resonance. Pace decelerates. Warmth replaces urgency.
• Eyes: Intensity softens from piercing to welcoming. Same commitment, gentler quality.
• Psychological intent: "I am a guide who has walked this path and is genuinely pleased to lead you through it."

Deep Dive/Value Delivery:
• Physiological: Dynamic movement within the frame. Spatial changes at major beat transitions (lean in, shift weight, forward/back). Facial expression HIGHLY CONGRUENT with content — no neutral face during complex explanation.
• Vocal: Gear 3. Modulate all four pillars: Pitch (high=inquiry/excitement, low=gravity), Pace (slow=comprehension, fast=momentum), Projection (loud=climax, whisper=secret), Pause (strategic silence creates anticipation).
• Pattern interrupts: implement at least one vocal gear shift or proximity change per 30-60 seconds.
• Psychological intent: "I am actively attempting to read their comprehension and adjusting to ensure they follow the logic."

Emotional Pivot/Storytelling:
• Physiological: Break eye contact — look slightly up-left as if accessing a real memory. Hold 0.3-1.0s. Return to lens for emotional payoff. Posture softens. Micro-expressions replace large gestures.
• Vocal: Gear 1. Quiet, intimate, breath-heavy. Pregnant pause 2-3 seconds after key reveals.
• Psychological intent: "I am trusting the audience with something sensitive. I am a vulnerable peer, not an expert authority."
• Story-specific: The specific detail (name, place, what was said) is not remembered — it is RE-LIVED in real time. The face responds to the memory as it forms, not to the script.

CTA:
• Physiological: Return to 10% forward lean. Break any relaxed storytelling posture. Re-establish commanding screen presence. Gestures become direct and literal — physically point to where viewer needs to look/click.
• Vocal: Gear 4. Confident, capable, overflowing with genuine enthusiasm. Never desperate. Volume increases slightly. Pace brisk — too slow gives time to click away. Urgency-based phrasing but authentic.
• Psychological intent: "I am not asking for a favor. I am providing the final piece of their solution. They would be leaving value on the table by not taking this step."
• CRITICAL: Any trace of selling/pitching energy detected by UHNWI viewer = immediate abandonment. Must read as authentic invitation, not commercial close.

EYE CONTACT MASTERY:
• Commit to the exact center of the lens glass — not the screen, flip monitor, or viewfinder
• Visualize a specific demographic avatar (one ideal viewer) sitting directly behind the glass
• Self-view monitor elimination: any visible self-view monitor guarantees the brain will break eye contact to self-monitor; remove it from the environment entirely
• Teleprompter technique: 3-4 feet back from camera; narrow text column directly over lens; eliminates tennis-match eye movement
• BREAKING eye contact is ONLY authentic during Storytelling/Emotional beats — signals memory access, not distraction

SINGER'S FORMANT & VOCAL STAMINA:
Diaphragmatic breathing is absolute prerequisite — never strain from upper vocal cords (causes thinning and fatigue in long sessions). The Singer's Formant is the resonant frequency cluster (2500-4000Hz) that allows the voice to cut through without volume increase — this is what "carrying power" means. Train by: lip trills, tongue trills, straw phonation (daily), double octave arpeggios. Chest voice (authority, trust, warmth) vs. head voice (excitement, vulnerability, intimacy) — specify which is called for per scene.

MISTAKE MANAGEMENT:
When an error occurs: pause, take one full diaphragmatic breath, leave 2-3 seconds of blank audio space for editor, pick up from one sentence before the error. Maintain performance energy through the gap — the editor needs the beat, not the restart. Never break character significantly. Trust the edit.

BEAT STRUCTURE (Stanislavski per scene):
Opening State (inherited from previous scene exit) → Building Action (scene rises toward gravity center) → Gravity Center (the single moment everything builds toward) → Resolution (breathing from the gravity center) → Exit State (deposits into next scene opening). The gravity center is identified in split_logic. Slow delivery for it — everything before it constructs, everything after breathes.
`;

// ============================================================
// API Key management
// ============================================================
let userApiKey: string | null = null;
export const setApiKey = (key: string) => { userApiKey = key; };

const getAI = () => {
  if (!userApiKey) throw new Error('API Key not set. Please provide your Google Gemini API Key.');
  return new GoogleGenAI({ apiKey: userApiKey });
};

const fileToBase64 = (file: File | Blob): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload  = () => resolve((reader.result as string).split(',')[1]);
    reader.onerror = reject;
  });

// ── Role-specific duration caps ──────────────────────────────
// VEO 3.1 handles extended scenes well; these roles need more room.
const ROLE_MAX_SECONDS: Record<string, number> = {
  'Framework':         12.0,
  'Action Framework':  12.0,
  'Storytelling':      12.0,
  'Case Study':        12.0,
  'Perspective Shift': 10.0,
  'Closing':           10.0,
  'Objection Handler': 10.0,
  'Value Delivery':     8.0,
  'Insight Reveal':     8.0,
  'Market Intelligence':8.0,
};
const DEFAULT_MAX_SECONDS = 8.0;
const getMaxSeconds = (role: string) => ROLE_MAX_SECONDS[role] ?? DEFAULT_MAX_SECONDS;

// ── Phoneme-explicit script annotation ───────────────────────
// Wraps emphasis words in *markers* and inserts [PAUSE-Xs] from the pause_map.
// These prosody anchors give VEO concrete hooks for speech synthesis.
function buildAnnotatedScript(
  scriptText:    string,
  emphasisWords: string[],
  pauseMap:      string[]
): string {
  let out = scriptText;

  // Emphasis markers — *word* signals phonemic completeness + slight duration
  for (const word of emphasisWords) {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replace(new RegExp(`\\b(${escaped})\\b`, 'gi'), '*$1*');
  }

  // Pause markers — insert [PAUSE-Xs] at the correct position
  for (const pause of pauseMap) {
    const durM  = pause.match(/(\d+\.?\d*)\s*s/i);
    if (!durM) continue;
    const dur   = durM[1];
    const afterM  = pause.match(/after\s+['"]([^'"]+)['"]/i);
    const beforeM = pause.match(/before\s+['"]([^'"]+)['"]/i);
    if (afterM) {
      const t = afterM[1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      out = out.replace(new RegExp(`(${t})`, 'i'), `$1 [PAUSE-${dur}s]`);
    } else if (beforeM) {
      const t = beforeM[1].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      out = out.replace(new RegExp(`(${t})`, 'i'), `[PAUSE-${dur}s] $1`);
    }
  }
  return out;
}

// ============================================================
// FUNCTION 1 — Analyze Reference Video
// ============================================================
export const analyzeReferenceVideo = async (videoFile: File): Promise<ReferenceAnalysis> => {
  const base64Data = await fileToBase64(videoFile);

  const prompt = `
You are operating at the convergence of three elite disciplines: Oscar-level performance coaching, world-class vocal science, and the precise technical requirements of AI video generation at photorealistic fidelity. This video is the DNA source for everything. Every observation you make here will directly determine whether the generated video looks like footage of a real human being or like AI. Precision is everything.

CRITICAL EXTRACTION PRIORITY NOTE: The character appearing in this reference video will NOT be reproduced in the generated video. A NEW character — whose identity is defined entirely by separately uploaded photos — will perform the new script. Your PRIMARY extraction focus is therefore: (1) PERFORMANCE DNA — how this person speaks, moves, pauses, emphasizes, and performs; their acting style, delivery patterns, energy, mannerisms, and vocal qualities; (2) VISUAL STYLE — the lighting setup, camera language, environment, and production quality. The character appearance you extract (face, wardrobe, skin) will be used only as a secondary reference for acting context — the new character's photos override all physical appearance.

═══════════════════════════════════════════════════════════
PHYSICAL IDENTITY — portrait painter precision (reference context only):
═══════════════════════════════════════════════════════════
Skin: Not "fair" or "medium" — the exact luminosity, the subsurface translucency where light penetrates before reflecting back, the micro-texture of actual human skin (individual pore character, the slight relief of pore rims catching the key light), the specific way this face's geometry creates specular highlights on the cheekbones and forehead, the warm amber undertone in the nasolabial folds where scattered light emerges.

Advanced skin physics — extract these four layers from the video:
FRESNEL EFFECT: At glancing angles (jaw edge, temples, lateral cheekbone, orbital rim), how much does the skin become more specular — wider, brighter highlights — vs. the diffuse quality of normally-incident zones (forehead center, nose bridge)? The angle-dependent reflectance is what makes the face genuinely three-dimensional.
SEBUM DIFFERENTIAL: Does the T-zone (forehead, nose, chin) show measurably higher specular return than the lateral cheeks — a mildly shinier quality vs. the more matte diffuse cheek planes?
VELLUS HAIR: Under close key light, is there a soft luminous halo of fine vellus facial hair visible at the cheekbone edge, jaw perimeter, or hairline? Not stubble — a barely-there translucent bloom.
DYNAMIC SKIN: As the jaw moves, does the skin over the masseter and mentalis region show micro-deformation — the biological stretch and compress of real flesh on jaw movement?

Bone structure: The jaw's exact terminus geometry — sharp/soft/rounded. The brow prominence above the orbits. The cheekbone plane's angle to the lens. The specific depth of the orbital ridge — does it create shadow over the eye? These are the structures of authority that read in the first 0.3 seconds.

Hair: Not "dark brown" — the exact chestnut/slate/warm-espresso that exists here. How it catches specular light at the crown. How much weight it has — does it move? Where the light creates a rim glow vs. where it absorbs. Individual hair strands visible at the perimeter.

Wardrobe: Not "suit" — the fabric weight (lightweight wool / medium-weight cashmere), the collar fall, the specific shade under this light, what it signals to a billionaire watching. Restraint? Earned taste? Precision?

Distinguishing features: The micro-asymmetries that make this face real. A slightly higher left brow. A specific shadow at a particular jaw angle. A scar, mark, or characteristic. These imperfections are what separate a photorealistic rendering from synthetic perfection.

═══════════════════════════════════════════════════════════
VOICE DNA — vocal coach at elite level:
═══════════════════════════════════════════════════════════
Describe the voice as a physical material — aged bourbon, brushed titanium, river stone — specific enough that a vocal coach who has never heard it could reproduce it.

Placement: Where does this voice physically live? Chest cavity (lower register, felt before heard), throat/larynx (mid-placement, forward energy), or mask/sinus (resonance at the front of the face, forward-projected). What percentage chest vs. mask?

Pace architecture: Describe exactly how pace shifts between modes — concept explanation (slower, constructed), storytelling (variable, alive), data delivery (precise, each number its own space), direct address to camera (intimate, the slowest pace).

Sentence endings: Does the voice fall at periods (authority — never seeking approval) or rise (uncertainty)? On the hardest falls, does the voice drop below baseline or simply stop ascending?

Silence behavior: Confident silence (inhabits the pause like it belongs there) or nervous silence (fills gaps before they fully form)? What is the maximum silence this person allows before it becomes weight?

The one vocal quality that makes an affluent real estate investor lean toward the screen — name it precisely.

US GENERAL AMERICAN ACCENT ASSESSMENT: Does this voice have General American characteristics (fully rhotic, Midwestern neutral, no regional coloring)? Or does it have accent characteristics that will need to be neutralized in the VEO output? Note any regional or international accent features so the VEO prompt can override them toward clean US General American.

Vocal placement and lip activity: Forward-placed voices produce more visible lip movement. Does the speech energy sit behind the teeth or at the lips? This determines how much articulation is visible on screen.

═══════════════════════════════════════════════════════════
MOUTH & ARTICULATION DNA — critical for AI lip sync:
═══════════════════════════════════════════════════════════
This section is the most technically critical for AI video generation. Describe with anatomical precision:

Resting lip geometry: The natural inter-lip gap (1-2mm? 4-5mm?), the specific curl of the upper lip, whether the lower lip is slightly forward of the upper, the commissure angle at rest (slightly down = seriousness; neutral = composure; slightly up = warmth).

Jaw mobility in speech: Does this person speak with a high jaw (minimal opening, 3-5mm on stressed vowels) or a low jaw (wide opening, 10-15mm on open vowels)? Does the jaw move quickly or does it lag?

Bilabial completeness: On /p/, /b/, /m/ sounds — do the lips make FULL contact (complete bilabial closure, clean release) or approximate contact (near-closure, blurred release)? Is there a characteristic compression just before the release?

Fricative precision: On /f/, /v/, /th/ — is the lip-to-teeth geometry precise and consistent, or relaxed and approximate? Forward or recessed tongue on /th/?

Sibilant character: Are /s/, /z/ crisp and forward-placed (tongue tip near upper teeth), or slightly lisped, or slightly retracted? Does the jaw assist or stay neutral on sibilants?

Pre-speech behavior: The exact physical sequence in the 0.3-0.5 seconds before the first syllable of any significant statement. Does the mouth open before the voice activates? Is there a visible inhale? A lip separation with a slight pull of the lower lip? A specific jaw-drop pattern?

Inter-word mouth state: What does the mouth do between words — does it return to a rest position (near-closed), hold the previous phoneme's shape, or remain slightly open throughout?

Breath visibility: Is the chest rise visible before phrases? Can the inhale be heard on the audio? Does the upper chest move or the diaphragm (lower chest)?

═══════════════════════════════════════════════════════════
ON-CAMERA AUTHORITY — YouTube-specific analysis:
═══════════════════════════════════════════════════════════
Natural resting expression through the lens — what does it communicate in the first 0.3 seconds to a viewer who has seen ten thousand faces?

Intimacy technique: Does this person speak TO the camera (creating a felt conversation) or AT the camera (a lecture posture)?

Eye contact: The blink rate and pattern (slow deliberate blinks = authority; rapid blinks = anxiety). Does the gaze hold through pauses or release on the outbreath? What does the eye do in the 0.2 seconds between thoughts?

The "intellectual engagement" tell: What does the face do in the half-second before revealing something important?

Gesture vocabulary: 3-5 natural gestures with the precise thought that triggers each:
- Authority gestures (palm-down, steeple, precision pinch)
- Openness gestures (open palm, spread hands)
- Building gestures (hands constructing in air)
- Emphasis gestures (index point, single finger lift)

The lean-in: What triggers it? How does the body load weight into the forward movement?

High-status stillness: Is silence filled with motion or inhabited with composure? What does the body do when the voice is not speaking?

The ONE thing this person does on camera that makes a UHNWI viewer think "this person is worth my time" — name it precisely.

═══════════════════════════════════════════════════════════
DELIVERY PATTERNS — thought leadership specific:
═══════════════════════════════════════════════════════════
Opening behavior: precisely what happens in the first 0.5 seconds of screen presence.
Build to insight: fast acceleration or slow deliberate construction?
Data and number delivery: what register, pace, physical attitude?
Storytelling shift: how does the body change? How does the voice change?
The "intellectual generosity" moment: where they give real value, how they physically signal it.
The "moment before" archetype: what is the face/body state in the 0.5 seconds before any important statement begins.

═══════════════════════════════════════════════════════════
VISUAL STYLE — the world they inhabit on camera:
═══════════════════════════════════════════════════════════
Lighting: key light direction and quality (hard/soft, beam angle), fill quality and depth, estimated key-to-fill ratio (3:1 gives moderate shadow depth; 6:1 creates dramatic authority contrast), color temperature in Kelvin.
Camera language: preferred framings with specific camera-to-subject distances, movement style, estimated focal length (which creates specific face geometry — 50mm is natural, 85mm is flattering compression, 35mm has slight distortion), angle tendency.
Background: exact tones, depth, gradient, what it does NOT say.

═══════════════════════════════════════════════════════════
FRAME LIBRARY — 25-35 peak performance moments:
═══════════════════════════════════════════════════════════
Categories to identify for UHNWI YouTube thought leadership:
- INTELLECTUAL AUTHORITY: the look of genuine expertise — calm, grounded, immovable
- WARM PEER-TO-PEER: speaking as an equal — zero hierarchy in either direction
- INSIGHT DELIVERY: face and body at the moment of revealing something important
- GENUINE AMUSEMENT: real reaction to something clever or counter-intuitive
- TRANSITIONAL THINKING: the face between thoughts — natural, human, unperformed
- PEAK CONVICTION: absolute certainty radiating from composure, not volume
- ACTIVE LISTENING POSTURE: present, still, completely receptive
- PRE-SPEECH MOMENT: the precise face and body state just before a significant statement begins

For each frame: include mouth_state — what the mouth is doing at this exact moment (resting, mid-word, between phrases, beginning speech, ending speech).

Return complete JSON matching exactly this structure:
{
  "video_title": "string",
  "video_summary": "string",
  "total_duration": "string",
  "character": {
    "appearance": "string — portrait-level: skin physics, bone structure, how light sculpts this specific face",
    "wardrobe": "string — fabric weight, drape, color precision, what it signals",
    "age_range": "string",
    "gender": "string",
    "build": "string",
    "hair": "string — color precision, texture, weight, light interaction",
    "skin_tone": "string — specific, not generic",
    "distinguishing_features": ["string — the micro-asymmetries and marks that make this face real"],
    "mouth_dna": {
      "rest_position": "string — inter-lip gap, jaw angle, lip tension, commissure angle at rest",
      "articulation_style": "string — forward/internal, labial activity level, overall mouth mobility",
      "jaw_openness": "string — range in mm on open vowels vs closed vowels, speed of jaw movement",
      "pre_speech_behavior": "string — the exact 0.3-0.5s sequence before any first syllable",
      "consonant_character": "string — bilabial completeness, fricative precision, sibilant placement",
      "breath_visibility": "string — chest rise visibility, inhale audibility, diaphragm vs. upper chest"
    },
    "voice": {
      "texture": "string — evocative material description",
      "pitch": "string",
      "pace_range": "string",
      "energy_baseline": "string",
      "accent": "string — geographic precision",
      "placement": "string — chest/mask ratio, forward or back-placed, lip activity implication",
      "qualities": ["string"]
    },
    "acting_style": {
      "persona_summary": "string",
      "default_expression": "string",
      "mannerisms": ["string"],
      "signature_gestures": ["string — gesture name + the specific thought that triggers it"],
      "eye_behavior": "string — blink rate, gaze-hold technique, eye behavior between thoughts",
      "body_language_patterns": ["string"],
      "emotional_range": "string",
      "transition_style": "string",
      "moment_before_archetype": "string — the precise face and body state in the 0.5s before any important statement"
    },
    "delivery_patterns": {
      "hook_technique": "string",
      "value_delivery_technique": "string",
      "cta_technique": "string",
      "pause_patterns": ["string"],
      "emphasis_method": "string",
      "pacing_strategy": "string"
    }
  },
  "visual_style": {
    "primary_location": "string",
    "lighting": {
      "key_light": "string — direction, quality, beam character",
      "fill_light": "string — quality, depth",
      "color_temperature": "string — in Kelvin where estimable",
      "shadows": "string — key-to-fill ratio and shadow character",
      "mood": "string"
    },
    "color_palette": "string",
    "background": "string — exact tones, depth, gradient",
    "props": ["string"],
    "atmosphere": "string",
    "camera_language": {
      "preferred_framings": ["string — with estimated camera-to-subject distances"],
      "movement_style": "string",
      "lens_characteristics": "string — focal length estimate and its effect on face geometry",
      "angle_tendency": "string"
    },
    "visual_style_summary": "string"
  },
  "frame_library": [
    {
      "timestamp": "MM:SS.S",
      "description": "string — specific physical position and what the body is doing",
      "expression": "string — precise emotional quality, not a generic label",
      "energy_level": 7,
      "body_position": "string",
      "mouth_state": "string — lip position, jaw angle, in-word / between-words / at-rest / breath",
      "suitability_tags": ["Hook"]
    }
  ],
  "performance_summary": "string — the precise, irreplaceable quality that makes this performer uniquely persuasive to a UHNWI audience"
}
`;

  const ai = getAI();
  let lastErr: any;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: MODEL_TEXT_ELITE,
        contents: [{ role: 'user', parts: [{ text: prompt }, { inlineData: { mimeType: videoFile.type, data: base64Data } }] }],
        config: { responseMimeType: 'application/json', thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
      });
      const raw = (response as any).text ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '';
      const json = safeJsonParse(raw, 'analyzeFullVideo');
      return json.referenceAnalysis?.character ? json.referenceAnalysis : json;
    } catch (err) {
      lastErr = err;
      if (attempt < 3) await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt - 1)));
    }
  }
  throw lastErr;
};

// ============================================================
// FUNCTION 2 — Segment Script
// ============================================================
export const segmentScript = async (
  newScript: string,
  referenceAnalysis: ReferenceAnalysis
): Promise<ScriptSegmentation> => {

  // Default caps — extended roles use their own limits below
  const MAX_WORDS   = 12;   // strict 95 WPM UHNWI pace math: 7.0s usable × 95/60 = 11 words max
  const MAX_SECONDS = 8.0;  // default; Framework/Storytelling/Case Study get 12s

  const prompt = `
You are a world-class YouTube director, Stanislavski-trained performance architect, script editor, and phonemics consultant. You are building an elite thought-leadership video for affluent real estate investors — experienced capital allocators who hold income-producing asset portfolios, think in IRR, equity multiples, cap rates, and risk-adjusted returns, and who have been pitched by everyone. You are building the directing blueprint that will govern both the human performance AND the AI generation of that performance in VEO 3.1. The presenter speaks as a peer — one experienced principal to another — never from a stage, always from the deal table.

CRITICAL MINDSET — THE VIDEO IS A SINGLE UNIT, NOT A SEQUENCE OF CLIPS:
Before you touch a single scene, read the entire script from first word to last. Hold the complete emotional and intellectual journey in your mind. The viewer experiences this as a continuous 60-120 second conversation — not as a playlist of short videos. Every segmentation decision must serve the whole. A scene is not a unit of content; it is a moment in a human being's experience of being persuaded of something true. The final scene is only as powerful as every scene that preceded it. Design the video the way a great novelist designs a chapter: each page earns the next; the last page is inevitable only because of everything before it.

YOUR THREE JOBS:
1. Generate a MASTER DIRECTING VISION governing every scene
2. Segment the script into precisely-timed VEO 3.1 scenes with elite acting blueprints
3. Give each scene a through-action, moment_before, and lip_sync_blueprint — the three layers that separate human-quality lip sync from mechanical generation

VEO GENERATION CONTEXT — DURATION OPTIMIZATION:
These scenes will be generated individually in VEO 3.1. Based on empirical testing: 7-8 seconds is VEO's optimal generation window — enough time for natural pre-speech behavior, deliberate UHNWI-pace delivery, pauses, and post-speech settle. Strongly prefer 7-8s for standard roles. For extended roles (Framework, Storytelling, etc.), use the full window they need. Scenes shorter than 5s are technically more challenging for VEO — if a short scene is necessary, design it with extreme deliberateness: fewer words, slower pace, long moment_before, generous post-speech settle. Minimum 4.0 seconds.

══════════════════════════════════════════════════
⚠ VEO 3.1 LIP SYNC — THE PRIMARY DESIGN CONSTRAINT (READ BEFORE EVERYTHING ELSE)
══════════════════════════════════════════════════

These scenes are not filmed — they are generated by VEO 3.1, an AI model that synthesizes speech and lip movement simultaneously. Lip sync quality is THE #1 failure mode. A perfect prompt with poor lip sync has completely failed. Every segmentation decision below must pass the lip sync filter first.

WHY LIP SYNC FAILS IN VEO 3.1 — THE FOUR KILLERS:
1. TOO MANY WORDS: When word count exceeds the WPM budget, VEO compresses delivery. The jaw cannot open fully on vowels because it is already moving to the next consonant. Result: mouth movements are small, mechanical, visibly AI-generated. This is the most common failure.
2. SCENE BREAKS MID-SENTENCE: When a scene begins mid-clause, VEO has no natural speech onset state. The mouth starts from neutral — geometrically wrong for mid-clause speech. The first word of a mid-clause scene almost always has incorrect lip sync.
3. PHONEMIC DENSITY: Dense consonant clusters (/str/, /nkr/, /spl/), multiple consecutive sibilants (/s/+/s/ or /z/+/s/), or back-to-back labial contacts (/m/+/b/) create sequencing overload in VEO's synthesis. The result: blurred transitions where individual phonemes lose their geometry.
4. MISSING ONSET GEOMETRY: When the moment_before does not specify the exact mouth position for the scene's FIRST phoneme, VEO generates an arbitrary starting position. The first 0.2 seconds of speech is almost always wrong.

THE LIP SYNC FIX — MANDATORY FOR EVERY SCENE:
A. WORD COUNT IS LAW: Use the strict math-derived limits below. Never exceed them — not even by one word.
B. SENTENCE BOUNDARY BREAKS ONLY: Break scenes ONLY at complete sentence endings (full stop, exclamation mark, question mark). Mid-clause breaks are prohibited.
C. FIRST PHONEME GEOMETRY: The moment_before must specify the exact mouth state for the first phoneme of the first word — jaw position in mm, lip geometry (closed/bilabial/spread/neutral-open), and whether the onset is voiced or unvoiced.
D. PHONEMIC DENSITY CONTROL: Rate every scene's lip sync risk (low/medium/high) based on phonemic complexity. High-risk scenes need extra word count reduction and slower WPM.
E. CO-ARTICULATION DESIGN: Design scenes so adjacent words blend naturally — no word boundaries with back-to-back stops that VEO renders as double pops.

══════════════════════════════════════════════════
⚠ SCENE BREAK LAW — VIOLATIONS CAUSE LIP SYNC FAILURE
══════════════════════════════════════════════════

LAW 1 — SENTENCE COMPLETENESS: Every scene must contain one or more COMPLETE sentences ending with a full stop (.), question mark (?), or exclamation mark (!). No exceptions.
LAW 2 — NO MID-CLAUSE BREAKS: Never break at a comma, semicolon, colon, or mid-clause position. The mouth has no natural reset at these points. VEO starts the next scene from neutral — wrong.
LAW 3 — EM-DASH EXCEPTION: A dramatic em-dash (—) may be a scene break ONLY if the next scene's first word begins with a bilabial (/b/, /m/, /p/) or open vowel — these allow VEO to start from a natural mouth-closed or open position.
LAW 4 — FIRST WORD PREFERENCE: Whenever the script allows, the first word of a scene should begin with: (a) a bilabial stop (/b/, /m/, /p/) — gives VEO a closed-lips starting state that naturally opens into the vowel; or (b) a voiced stop (/d/, /g/) — defined onset geometry. Avoid starting scenes with sibilants (/s/, /z/, /ʃ/) — these have no defined onset geometry and frequently produce first-word lip sync artifacts.
LAW 5 — CLOSING PHONEME: The last word of a scene should end with a stop consonant (/t/, /d/, /k/, /p/) that gives VEO a clean final closure, or a full vowel followed by a nasal (/m/, /n/) that allows the mouth to settle naturally into the post-speech position.

══════════════════════════════════════════════════
⚠ WORD COUNT HARD LIMITS — DERIVED FROM 95 WPM UHNWI PACE MATH
══════════════════════════════════════════════════

Base rate: UHNWI authority delivery = 95 WPM (deliberate, measured, every word placed).
Pre-speech warm-up: 0.5s. Post-speech settle: 0.5s. Each strategic pause: 0.5s.

STANDARD ROLES:
• 8s scene: usable speech = 7.0s → 11.1 words → HARD MAX: 12 words (zero pauses) / 10 words (one pause)
• 7s scene: usable speech = 6.0s → 9.5 words → HARD MAX: 10 words
• 6s scene: usable speech = 5.0s → 7.9 words → HARD MAX: 8 words
• 5s scene: usable speech = 3.7s → 5.9 words → HARD MAX: 6 words
• 4s scene: usable speech = 2.7s → 4.3 words → HARD MAX: 4 words

EXTENDED ROLES (deliberate slower pace, 85 WPM):
• 12s: usable speech = 11.0s → 15.6 words → HARD MAX: 16 words
• 10s: usable speech = 9.0s → 12.75 words → HARD MAX: 13 words (Perspective Shift/Closing/Objection Handler)

EMPHASIS WORD TAX: Each emphasis word requires 25% more time than a normal word. For every 2 emphasis words, reduce the word count budget by 1.

If the original script line exceeds these limits, you MUST trim it. Preserve meaning — remove connective filler, redundant qualifiers, and padding. Never touch the core idea, the key number, the named individual, or the insight. The trimmed script_text is the deliverable; preserve original in original_script_text.

LIP SYNC RISK RATING (assign to every scene):
• LOW: ≤8 words, mostly open vowels and stop consonants, no sibilant clusters
• MEDIUM: 9-11 words, some sibilants or fricatives, ≤2 consonant clusters
• HIGH: 12+ words, multiple sibilants, dense consonant clusters, or back-to-back labial contacts — requires maximum word count reduction and extra phonemic attention in the prompt

═══════════════════════════════════════════════════
PART 1 — MASTER DIRECTING VISION
═══════════════════════════════════════════════════

Before touching a single scene, think as a master director across the full arc.

VOICE FINGERPRINT: One locked vocal description from the Presenter DNA. The exact acoustic character — same register, same placement, same baseline authority — that appears in every scene identically. One vivid sentence that a vocal coach who has never heard this voice could use to find it.
e.g. "Aged bourbon poured slowly — warm chest resonance, unhurried authority, every sentence landing with the weight of evidence behind it."

ENERGY ARC MAP: The energy architecture as a narrative physics problem. Valleys make peaks feel earned. Silence makes arrivals important. Map scene-by-scene as: "Scene 1: 8/10 (arrival) → Scene 2-3: 6/10 (generous depth) → Scene 4: 5/10 (intimacy valley — energy debt accumulating) → Scene 5: 9/10 (peak conviction — debt paid) → Scene 6: 5/10 (measured landing)". The valley-to-peak ratio is the engine of sustained attention.

CHARACTER THROUGH-LINE: The one persona constant that makes this presenter recognizable and consistent from frame one to frame last. The trait the viewer could articulate after watching.

VISUAL ANCHOR: The locked visual world established in Scene 1. Background depth, lighting signature, color temperature, framing language, camera-to-subject distance. Lock this with enough specificity that VEO can recreate it scene by scene.

MASTER THROUGH-ACTION (Stanislavski super-objective): The active verb phrase governing the entire video's psychological trajectory — what this video is doing to the viewer's mind across all scenes.
NEVER: "to present", "to explain", "to describe". ALWAYS: a transformation.
e.g. "To make this viewer understand that the risk they fear is imaginary and the opportunity they're missing is real — and hand them the conviction to act."

PRESENTATION PERSONA: The UHNWI-appropriate archetype inhabiting this video.
e.g. "The world-class private advisor who speaks to principals as peers — never performing, always transmitting, treating the viewer's time as the most valuable thing in the room."

SILENCE RULE (ABSOLUTE): Zero music. Zero audio effects. Zero ambient sound. Zero subtitles. Voice only, in complete acoustic silence. Every scene, no exceptions.

═══════════════════════════════════════════════════
YOUTUBE DIRECTING BIBLE — RETENTION ARCHITECTURE (MANDATORY):
═══════════════════════════════════════════════════
${YOUTUBE_DIRECTING_BIBLE}

═══════════════════════════════════════════════════
ACTING BIBLE — VOCAL GEAR & PERFORMANCE SYSTEM (MANDATORY):
═══════════════════════════════════════════════════
${ACTING_BIBLE_DIRECTIVES}

BIBLE INTEGRATION REQUIREMENTS — MANDATORY FOR EVERY SCENE:
1. RETENTION TARGET: Assign each scene a retention_target_percent based on its position in the video's retention curve (90%/80%/75%/70%/65%/40% per curve above).
2. VOCAL GEAR: Assign each scene a vocal_gear (1, 2, 3, or 4) based on scene type and energy requirements per the Acting Bible.
3. BIBLE SCENE TYPE: Classify each scene as one of: "Hook" / "Setup" / "Deep Dive" / "Emotional Pivot" / "CTA" — this maps to the Acting Bible's scene-by-scene directive system.
4. AMPLIFIED SELF: Every scene's energy_level must account for the 10% camera absorption principle. A 7/10 natural energy must be directed at 8/10 to read correctly on screen.
5. PATTERN INTERRUPT MAPPING: Flag scenes that serve as pattern interrupts. Ensure at least one occurs every 30-60 seconds of real viewing time across the full video.
6. CLICK CONFIRMATION: The first scene (Hook) MUST apply PVSS formula. Assign the specific Hook archetype used (Fortune Teller / Investigator / Contrarian).
7. LIGHTING DIRECTION: Specify three_point_lighting_direction (Key angle/quality, Fill ratio, Rim usage) per scene based on emotional tone and scene type.
8. FOCAL LENGTH: Assign lens_focal_length (16-24mm / 35-50mm / 85mm+) per scene based on psychological purpose per the Directing Bible.
9. CTA PLACEMENT: If video contains a CTA scene, verify it lands immediately after the highest-value delivery scene in the sequence (reciprocity pause law).
10. GRAVITY CENTER: Identify the gravity_center_word — the single word/phrase that is the entire reason the scene exists, which receives maximum deceleration and lowest pitch.

EMOTIONAL DEPOSIT/WITHDRAWAL LEDGER:
Each scene makes an emotional transaction with the viewer. DEPOSITS: trust (through specificity and honesty), curiosity (through open loops or surprising claims), authority (through evidence-backed certainty), warmth (through peer-level generosity). WITHDRAWALS: attention budget (any scene that does not pay for itself), patience (padding, repetition, hedging), trust (any trace of performance over truth). The net emotional ledger across the full video must be deeply positive — the viewer finishes richer than they started. Design each scene with: emotional_deposit (what the viewer receives) and attention_cost (what focus it requires). No scene should cost more than it gives. Assign these as fields in each scene's continuity block.

RETENTION TECHNIQUE MAPPING — per scene, assign one primary technique:
SPECIFICITY_ANCHOR: A single hyper-precise detail (exact number, named individual, specific location) that makes the scene feel grounded in reality. Viewers stay for specificity because it signals authenticity.
KNOWLEDGE_GAP: The viewer realizes mid-scene they don't know something important — and the answer is coming. The gap is the retention mechanism; every second it remains open, the viewer stays.
PEER_RECOGNITION: A moment where the sophisticated viewer thinks "this person understands my world exactly" — a vocabulary choice, a professional assumption, a shared reference that signals insider-to-insider transmission.
PATTERN_VIOLATION: Something that contradicts what the viewer expected — a counter-intuitive claim, a data point against consensus — that makes the viewer recalibrate their mental model and stay to find out if they're right.
EARNED_REVELATION: The viewer has done enough intellectual work alongside the presenter that the insight feels discovered rather than delivered. The journey made the destination valuable.
Assign one of these five techniques as retention_technique per scene in the continuity object.

SCENE-TRANSITION ARCHITECTURE — design each cut as a retention decision:
The boundary between scenes is not where the words end — it is a constructed moment. For each scene: the EXIT STATE (what emotional/attentional state the viewer is in at the final frame) must be designed to create the ENTRY REQUIREMENT of the next scene. The expression carried from the final frame of one scene into the first frame of the next is the expression_inheritance — a residual emotional color that makes the performance feel humanly continuous rather than scene-by-scene reset. Map this for every scene pair: what expression quality exits → what that creates as the opening state of the next scene. Assign this as expression_inheritance in each scene's continuity block.

═══════════════════════════════════════════════════
PART 2 — SCENE SEGMENTATION (sub-8 second hard limit)
═══════════════════════════════════════════════════

THE AFFLUENT REAL ESTATE INVESTOR WATCHING ON YOUTUBE: Has underwritten hundreds of deals. Has been in every pitch meeting. Detects inauthenticity in 3 seconds — because they've been misled before and it cost them. Stays for: peer-level register that respects their experience, access to genuine deal-table thinking, market intelligence they couldn't get elsewhere, earned authority backed by specific evidence. Leaves for: any trace of hype replacing evidence, generic investment advice, condescension, wasted pace, anything that feels scripted rather than genuinely thought. Their professional instinct is to qualify information sources before trusting them. The first 10 seconds either earn their trust or confirm their skepticism. Design every scene to earn trust.

SCENE GRAVITY CENTERS: Every scene has one moment of maximum gravity — the single word, phrase, or silence that is the entire reason for this scene's existence. Everything before it builds. Everything after it breathes. Identify this gravity center in the split_logic for every scene.

ENERGY MOMENTUM PHYSICS:
- Low-energy (4-5/10) following high-energy (7-8/10) = contrast that makes the next peak feel earned
- Two consecutive scenes below 6/10 creates an energy debt — the following scene must exceed 8/10
- Each scene inherits energy from the previous and deposits energy into the next
- The outframe's final expression sets the inframe's opening state

RETENTION ARC (structure the full video around this):
1. Hook (first 30s): Credibility established + specific curiosity planted that only staying resolves
2. Immediate payoff: First value hit within 60 seconds — prove the opening promise
3. Deepening value: Each scene leaves the viewer measurably richer
4. Re-engagement peaks: Pattern Interrupt or Perspective Shift every 60-90 seconds
5. Cumulative authority: Each scene earns more trust than the one before
6. Generous close: Viewer leaves richer than they arrived — satisfied, not sold

═══════════════════════════════════════════════════
TIMING ENFORCEMENT — ABSOLUTE:
═══════════════════════════════════════════════════

HARD RULE: Every scene deliverable in ≤${MAX_SECONDS} seconds.

VEO 3.1 OPTIMAL DURATION WINDOW — 7-8 SECONDS:
Research-proven: VEO 3.1 achieves its highest-quality speech synthesis in the 7-8 second range. In this window, there is natural room for pre-speech behavior (0.3-0.5s), deliberate UHNWI-pace speech, pause moments, and post-speech settle — without any element feeling rushed. When designing scene lengths for standard roles, STRONGLY PREFER 7-8 seconds. Only go shorter when the script fragment genuinely cannot be stretched without losing authenticity. Never pad artificially — but pace and pauses are legitimate timing tools.

SHORT SCENE GUIDANCE (≤5.5 seconds):
When a scene must be short (3-5 words, clear standalone moment), design it with:
- SLOWER delivery pace than normal — every word receives more space
- A long pre-speech behavioral moment (0.6-0.8s of settled presence before first word)
- A generous post-speech settle (0.5-0.8s of the expression holding after last word)
- Fewer total words than the timing math alone suggests — the pauses and behavioral moments carry the scene
- The moment_before must be especially detailed — it's a larger proportion of this scene's total time
Minimum scene duration: 4.0 seconds (scenes shorter than this produce mechanical-onset issues in VEO).

TIMING MATH:
  - 8 seconds of speech = maximum ${MAX_WORDS} spoken words at UHNWI deliberate pace
  - Each pause ≈ 0.4-0.6s, reducing word budget
  - Short scenes (≤5.5s): subtract 1.3s from duration for warm-up/settle; remainder is speech window
  - Formula: (duration - 1.3s for short / 0.8s for standard - total_pause_seconds) × (100-130 WPM) = word budget

SCRIPT ADJUSTMENT AUTHORITY:
  ALLOWED: Remove connective filler, tighten redundant qualifiers, compress setup
  NEVER CHANGE: Core idea, key insight, named data, meaning, voice, register
  If trimmed: preserve original in original_script_text; count words precisely

AVAILABLE ROLES:
Core: Hook / Pattern Interrupt / Value Delivery / Social Proof / Bridge / Call to Action / Storytelling / Demonstration / Objection Handler / Open Loop / Closing
YouTube Thought Leadership: Insight Reveal / Framework / Case Study / Market Intelligence / Perspective Shift / Action Framework

ROLE TIMING PROFILES (VEO 3.1 lip-sync-optimised):
Standard roles (target 7-8s / HARD MAX per word count math above):
- Hook: 7-8s | max 11 words — impact over density; whitespace around the gravity center
- Value Delivery: 7-8s | max 12 words — one value per scene, not compressed value stacks
- Insight Reveal: 7-8s | max 11 words — the insight needs silence before AND after to land
- Market Intelligence: 7-8s | max 12 words — data delivered with space; each number its own beat
- Demonstration: 7-8s | max 12 words — crisp and specific; no qualifiers
- Call to Action: 7-8s | max 12 words — direct action language; forward vowels; strong onset
- Social Proof: 6-7s | max 10 words — understatement; specific numbers; zero performative energy
- Bridge: 6-7s | max 10 words — momentum carrier; warmth in the consonants
- Open Loop: 6-7s | max 10 words — incompletion at sentence end; suspended final pitch
- Pattern Interrupt: 4-6s | max 7 words — brevity IS the interrupt; short = surprising
- Storytelling: 8-12s | max 16 words (authentic memory requires space; pace is naturally slower)
- Framework: 8-12s | max 16 words (each component its own beat; no stacking)
- Action Framework: 8-12s | max 16 words (each action step receives equal space)
- Case Study: 8-12s | max 16 words (specific detail + the implication it carries)
- Perspective Shift: 8-10s | max 13 words (the conventional view + the turn need separate space)
- Objection Handler: 8-10s | max 13 words (pause before the response is a performance moment)
- Closing: 8-10s | max 13 words (the slowest scene; every word placed; weight before silence)

═══════════════════════════════════════════════════
ACTING BLUEPRINT — per scene:
═══════════════════════════════════════════════════

SCENE ESSENCE: One evocative metaphor — not what happens, the feeling. The emotional north star.
- Hook: "A door opens in a room where everyone thought the walls were solid."
- Insight Reveal: "A gem placed on a table, lit from within, needing no explanation."
- Framework: "An architect revealing the blueprint of something that took a decade to build."
- Market Intelligence: "A Bloomberg analyst who just caught a signal nobody else has seen."
- Perspective Shift: "The moment an optical illusion flips — you can never unsee it."
- Closing: "The end of a great conversation where both parties received more than they gave."

THROUGH-ACTION (Stanislavski active verb — the spine of the performance):
What the presenter is ACTIVELY DOING to the viewer's psychology in this scene — not what they say but what they intend to produce. A transitive verb targeting the viewer.
NEVER: "to explain" / "to describe". ALWAYS: a transformation.
- "To crack open the viewer's assumption and let certainty flood in."
- "To hand the viewer intellectual property they could not have built themselves."
- "To make the viewer feel they are the only person this is being said to."
- "To invite — not pressure — toward the next step."

CHARISMA QUALITY (required for every scene):
The specific magnetic quality that makes this scene worth watching even before the viewer consciously processes what is being said. One sentence, physically specific. Not "confidence" or "authority" — the precise felt quality of the performance.
Examples:
- "The settled certainty of someone who has seen this market move exactly as they said it would, and is watching it move again."
- "The private warmth of an advisor who is about to hand over the exact thing the viewer has been looking for without knowing what to call it."
- "The barely-contained intellectual pleasure of someone placing their best piece of thinking in front of the one person in the room who can receive it."
- "The composure of someone who has been asked this question a thousand times and finds it genuinely interesting every time because the asker always teaches them something."

SUBTEXT (what the face says beneath the words — required):
The face always says something more complex than the words. What is the secondary emotional layer running beneath the primary content of this scene?
Examples:
- "I have been where you are with this, and what I found changed my approach entirely. You will feel that shift."
- "This is not common knowledge. The fact that I'm giving it to you now means I consider you capable of using it."
- "I know you've been pitched this framing before by people who didn't understand it. What follows is different."
- "The conventional view on this is wrong in a specific, important way. I discovered that at cost. You receive the discovery for free."

LEAN-IN SIGNAL (the physical quality that makes the viewer incline toward the screen):
One physically specific direction for the quality that creates involuntary viewer engagement. Not "be charismatic" — the precise physical mechanism.

MOMENT BEFORE (Stanislavski pre-speech state):
The precise physical and psychological state in the 0.5 seconds before the first word arrives. This is what VEO renders as the pre-speech frame and what makes lip sync feel genuinely human.
Include: jaw position, lip state, breath state, internal psychological experience, eye contact quality.
e.g. "The jaw is relaxed, lips within 3mm of each other — not touching. The thought is completely formed. Eyes already engaged with the lens, the warmth of what is about to be said already visible in the orbital muscles. A chest expansion — quiet, 0.3 seconds — as the breath prepares. The lips part with almost no effort, and the first word arrives as if it was always going to."

EMOTIONAL CORE (UHNWI register — one phrase):
"Sovereign certainty" / "Intellectual generosity" / "Conspiratorial warmth" / "Earned authority" / "Calibrated conviction" / "Quiet revelation" / "Peer-level respect"

PHYSICAL SIGNATURE: ONE posture-state VEO holds and animates from.
"The stillness of a grandmaster who sees the board clearly."
"The forward lean of a mentor about to hand over a decade of learning."

GESTURES (max 2, UHNWI-appropriate, natural to this presenter's DNA):
Steeple / Open palm toward viewer / Precision pinch / Hands building in air / Single index / Slow deliberate lean-in

EXPRESSION: Written as the performer feels it from inside — not external description.
"The eyes already hold the answer to a question the viewer hasn't asked yet."

PAUSE MAP: Silence is authority. Map exactly where silence lives and what it does.

LIP SYNC BLUEPRINT — PHONEMIC PRE-COMPUTATION (the technical layer for VEO):
For each scene's script_text, analyse phonemically at clinical precision. This data is the authoritative lip-sync reference injected directly into the VEO prompt.

PHONEMIC ANCHORS: Identify the 4-6 most visually demanding words. For EVERY anchor, you MUST include ALL FIVE elements:
1. WORD: the exact word as written in script_text
2. PHONEME: the IPA symbol and class (e.g. /æ/ = open front vowel; /b/ = voiced bilabial stop)
3. JAW OPENING: in mm at peak — 0mm(closed), 3mm(slight), 6mm(moderate), 10mm(full), 14mm(max)
4. LIP GEOMETRY: one of: bilabial-closed / bilabial-released / spread / rounded / neutral-open / dental-contact / neutral
5. VEO FAILURE MODE: the specific AI error for this phoneme class (e.g. "insufficient jaw travel on /æ/ — jaw stays at 5mm; vowel sounds closed and muffled")

Mandatory coverage: all /p/, /b/, /m/ words (bilabials must close to 0mm fully); all words with /æ/, /ɑ/, /aɪ/ (wide vowels need 10-14mm jaw); the scene's gravity center word (full anchoring regardless of phoneme); any word with consecutive sibilants or fricatives.

Example anchor: "'capital' → /æ/ peak: jaw 10mm, lips neutral-open, tongue fully depressed; /p/ onset: bilabial-closed 0mm, clean release into /ɪ/; VEO failure: jaw rarely reaches 10mm — vowel sounds closed. "'believe' → /b/: bilabial-closed 0mm → bilabial-released → /ɪ/ 3mm spread; VEO failure: lips approximate but don't seal — /b/ renders as /v/"

SPEECH ONSET PHONEME: The exact mouth state for the FIRST phoneme of the FIRST word of this scene. This is what VEO renders as the pre-speech frame onset. Format: "[word]: [phoneme] onset — jaw [Xmm], lips [geometry], [voiced/unvoiced], duration [Xs] before vowel opens"

JAW TRAVEL MAP: Min-to-max jaw travel for this script (in mm). List the 3 words with maximum jaw opening and the 3 words with minimum. Note any dramatic jaw excursion that must be rendered continuously (not as a jump cut).

LIP TENSION NOTES: High-labial (frequent visible lip contacts — bilabials, labiodentals) or low-labial (more internal alveolar/velar work)? Bilabial release character: crisp authority (fast clean pop) or warm deliberate (slightly longer contact before release)? Any asymmetry from this presenter's mouth_dna?

BREATH POINTS: Every breath event as a physical occurrence: before which word, duration (0.2-0.5s), type (visible chest expansion / silent diaphragmatic / audible soft intake), and whether it is heard in the audio.

CO-ARTICULATION NOTES: Flow character (continuous blend across word boundaries — organic human speech) or place character (each word discrete — more formal/deliberate)? List 2-3 specific word boundaries where co-articulation must be continuous (e.g. "the market" → /ðəmɑrkɪt/ flows without stop between "the" and "market"). This determines whether lip sync feels human or robotic.

═══════════════════════════════════════════════════
INPUTS:
═══════════════════════════════════════════════════
NEW SCRIPT:
"""${newScript}"""

PRESENTER DNA:
${JSON.stringify(referenceAnalysis?.character || {}, null, 2)}

FRAME LIBRARY:
${JSON.stringify(referenceAnalysis?.frame_library || [], null, 2)}

═══════════════════════════════════════════════════
RETURN COMPLETE VALID JSON — exactly this structure:
═══════════════════════════════════════════════════
{
  "total_scenes": number,
  "narrative_arc": "string — the full intellectual + emotional journey as a UHNWI viewer experiences it",
  "directing_vision": {
    "voice_fingerprint": "string — one vivid locked sentence governing every scene",
    "energy_arc_map": "string — scene-by-scene energy map with momentum physics",
    "character_through_line": "string — the one constant persona trait",
    "visual_anchor": "string — the locked visual world Scene 1 establishes",
    "through_action": "string — master active verb governing the video's full psychological arc",
    "silence_rule": "Zero music. Zero audio effects. Zero ambient sound. Zero subtitles. Voice only.",
    "presentation_persona": "string — the UHNWI presenter archetype"
  },
  "scenes": [
    {
      "scene_number": number,
      "role": "string",
      "title": "string — evocative 3-4 word title",
      "duration_seconds": number,
      "word_count": number,
      "script_text": "string — final text (trimmed if needed)",
      "original_script_text": "string — only if adjusted; else omit this field",
      "narrative_position": "string — e.g. Scene 2 of 8 — authority build, energy rising 7→8",
      "split_logic": "string — why cut here + gravity center + timing rationale",
      "emotional_tone": "string — precise feeling in a UHNWI viewer",
      "energy_level": number,
      "retention_target_percent": number,
      "bible_scene_type": "Hook | Setup | Deep Dive | Emotional Pivot | CTA",
      "gravity_center_word": "string — the single word/phrase that is the entire reason this scene exists",
      "hook_archetype": "string — only for Hook scenes: Fortune Teller / Investigator / Contrarian",
      "is_pattern_interrupt": false,
      "lip_sync_risk": "low | medium | high",
      "acting_blueprint": {
        "scene_essence": "string — one evocative metaphor, the north star",
        "through_action": "string — active transitive verb phrase: what the presenter does to the viewer",
        "emotional_core": "string — single dominant emotion, UHNWI register",
        "physical_signature": "string — ONE defining posture-state",
        "moment_before": "string — jaw position, lip state, breath state, psychological experience, eye contact quality in the 0.5s before first word",
        "charisma_quality": "string — the specific magnetic quality of this scene that creates involuntary viewer engagement; physically precise, not a label",
        "subtext": "string — what the face says beneath the words; the secondary emotional layer running beneath primary content; one sentence from inside the performance",
        "lean_in_signal": "string — the precise physical mechanism that makes the viewer's body incline toward the screen; specific and felt, not generic",
        "intention": "string — what the presenter wants the viewer to FEEL",
        "delivery_pace_wpm": number,
        "vocal_gear": 1,
        "vocal_gear_rationale": "string — why this gear for this scene per the Acting Bible",
        "forward_lean": false,
        "pregnant_pause_required": false,
        "amplified_energy_level": number,
        "speech_onset_phoneme": "string — FIRST phoneme of FIRST word: jaw Xmm, lip geometry, voiced/unvoiced, duration before vowel opens",
        "eye_contact_technique": "string — committed / break-for-memory / building-connection — with specific direction",
        "lighting_direction": "string — Key angle/quality, Fill ratio, Rim usage, background color",
        "focal_length": "string — 16-24mm / 35-50mm / 85mm+ with psychological rationale",
        "emphasis_words": ["string — 2-4 words of maximum weight"],
        "pause_map": ["string — where silence lives, with feeling and duration"],
        "energy_arc": "string — how energy moves within this scene",
        "mapped_mannerisms": ["string — natural to this presenter, max 2"],
        "mapped_gestures": ["string — UHNWI-appropriate, precisely described, max 2"],
        "expression_direction": "string — felt from inside, not described from outside",
        "body_direction": "string — motivated postural journey, thought-driven",
        "lip_sync_blueprint": {
          "phonemic_anchors": ["string — word → phoneme (IPA) + jaw Xmm + lip geometry + VEO failure mode"],
          "jaw_travel_map": "string — overall travel + peak-open and peak-closed words",
          "lip_tension_notes": "string — bilabial release character, labial activity level",
          "breath_points": ["string — before which word; duration; visible chest/silent intake"],
          "co_articulation_notes": "string — flow vs. place; organic blending character"
        }
      },
      "recommended_inframe": { "timestamp": "MM:SS.S", "rationale": "string" },
      "recommended_outframe": { "timestamp": "MM:SS.S", "rationale": "string" },
      "camera_direction": {
        "framing": "string — MCU / tight MCU / CU with camera-to-subject distance",
        "movement": "string — locked-off / imperceptible push-in / specific motivation",
        "angle": "string",
        "lens": "string — focal length equivalent and its face-geometry effect",
        "depth_of_field": "string — where focus falls and where it softens"
      },
      "continuity": {
        "enters_from": "string — exact energy + mouth state + posture arriving from previous scene",
        "exits_to": "string — how this scene's final state sets up the next scene's opening",
        "expression_inheritance": "string — the specific facial expression quality carried from the previous scene's final frame into this scene's first 0.3-0.5s; not the new scene's expression, but the residue of the previous one that organically dissolves into this scene's truth",
        "emotional_deposit": "string — what this scene gives the viewer: trust/curiosity/authority/warmth + the specific mechanism through which they receive it",
        "attention_cost": "string — what focus this scene requires from the viewer; must be meaningfully less than the deposit",
        "retention_technique": "string — SPECIFICITY_ANCHOR | KNOWLEDGE_GAP | PEER_RECOGNITION | PATTERN_VIOLATION | EARNED_REVELATION — one per scene with one-sentence rationale for why this technique fits this scene's position in the arc"
      }
    }
  ]
}

QUALITY CHECK BEFORE RETURNING — FAIL ANY SCENE THAT VIOLATES THESE:
LIP SYNC LAWS:
✗ FAIL if any standard scene has word_count > 12 — reduce script to fit
✗ FAIL if any extended role (Framework/Storytelling/Case Study/Action Framework) has word_count > 16
✗ FAIL if any extended role (Perspective Shift/Objection Handler/Closing) has word_count > 13
✗ FAIL if any scene breaks mid-sentence — every scene must end at a full stop (.) question mark (?) or exclamation (!)
✗ FAIL if moment_before does not include the speech_onset_phoneme: jaw position in mm, lip geometry, and onset type
✗ FAIL if any phonemic_anchor is missing any of the 5 required elements (word / phoneme / jaw mm / lip geometry / VEO failure mode)
✗ FAIL if lip_sync_risk is missing from any scene

TIMING LAWS:
✗ FAIL if any scene is shorter than 4.0s
✗ FAIL if any standard role exceeds 8.0s
✗ FAIL if any extended role exceeds 12.0s
✗ FAIL if short scenes (≤5.5s) have word_count > 6

CONTENT LAWS:
✓ script_text preserves 100% of the original meaning — if trimmed, original_script_text is populated
✓ moment_before includes jaw position, lip state, breath state, AND psychological experience — all four
✓ speech_onset_phoneme specifies the exact mouth state for the first phoneme of the first word
✓ through_action is a transitive active verb targeting the viewer — never "to explain" or "to describe"
✓ lip_sync_blueprint.phonemic_anchors references specific words from that scene's script_text
✓ All recommended_inframe and recommended_outframe timestamps exist in the provided frame library
✓ continuity.expression_inheritance describes a specific residual expression quality from the previous scene
✓ continuity.retention_technique is one of the five named techniques with rationale tied to this scene's arc position
✓ continuity.emotional_deposit names what the viewer receives + the specific mechanism through which they receive it
✓ Scene 1 continuity.expression_inheritance = "Opens from internal stillness — no residue; the presenter arrives fresh, settled, and already present before a single word"
✓ The full video narrative arc is coherent: no two consecutive scenes below 6/10 energy without a peak following; energy_arc_map accurately reflects all scene energy_levels
`;

  const ai = getAI();
  let lastErr: any;
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: MODEL_TEXT_ELITE,
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        config: { responseMimeType: 'application/json', thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
      });
      const raw = (response as any).text ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '';
      return safeJsonParse(raw, 'segmentScript');
    } catch (err) {
      lastErr = err;
      if (attempt < 3) await new Promise(r => setTimeout(r, 1000 * Math.pow(2, attempt - 1)));
    }
  }
  throw lastErr;
};

// ============================================================
// FUNCTION 3 — Extract Frame (Client-Side)
// ============================================================
export const extractFrameFromVideo = (videoFile: File, timestamp: string): Promise<Blob> =>
  new Promise((resolve, reject) => {
    const video  = document.createElement('video');
    const canvas = document.createElement('canvas');
    const ctx    = canvas.getContext('2d')!;

    video.preload     = 'auto';
    video.muted       = true;
    video.playsInline = true;

    const parts   = timestamp.split(':');
    let seconds   = 0;
    if (parts.length === 2)      seconds = parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    else if (parts.length === 3) seconds = parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    else                         seconds = parseFloat(timestamp);
    if (isNaN(seconds)) seconds = 0;

    const cleanup = () => { URL.revokeObjectURL(video.src); video.src = ''; };
    const timer   = setTimeout(() => { cleanup(); reject(new Error('Frame extraction timed out')); }, 15000);

    video.onloadedmetadata = () => {
      canvas.width      = video.videoWidth  || 1280;
      canvas.height     = video.videoHeight || 720;
      video.currentTime = Math.min(seconds, video.duration - 0.1);
    };

    video.onseeked = () => {
      clearTimeout(timer);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => {
        cleanup();
        if (blob) resolve(blob);
        else reject(new Error('Frame extraction failed'));
      }, 'image/jpeg', 0.97);
    };

    video.onerror = () => { clearTimeout(timer); cleanup(); reject(new Error('Video failed to load')); };
    video.src = URL.createObjectURL(videoFile);
  });

// ============================================================
// FUNCTION 3.5 — Generate Character Frame (2K · Hyper-Real)
//
// Strategy: Character photos define EVERYTHING (identity, setting,
// lighting, wardrobe). Pose reference provides body geometry only.
// Output: 2K (2048px) 16:9 photorealistic frame via imageConfig.
// SDK imageSize bug workaround: canvas upscale to 2048px if needed.
// Silent fallback to character's own photo if model unavailable.
// ============================================================

// ── 2K upscale guarantee ──────────────────────────────────────
// Works around the known @google/genai SDK bug where imageSize:'2K'
// is sometimes ignored for gemini-3-pro-image-preview. If the model
// returns a smaller image, we upscale it to 2048px width client-side
// using high-quality Lanczos-equivalent canvas interpolation.
const upscaleTo2K = (blob: Blob): Promise<Blob> =>
  new Promise(resolve => {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      const TARGET_W = 2048;
      if (img.naturalWidth >= TARGET_W) {
        // Already 2K or larger — return as-is
        URL.revokeObjectURL(url);
        resolve(blob);
        return;
      }
      // Upscale using two-pass progressive canvas scaling for quality
      const scale  = TARGET_W / img.naturalWidth;
      const finalH = Math.round(img.naturalHeight * scale);

      // Pass 1 — intermediate scale to 1.5× (reduces aliasing vs. direct jump)
      const midW = Math.round(img.naturalWidth * Math.sqrt(scale));
      const midH = Math.round(img.naturalHeight * Math.sqrt(scale));
      const c1   = document.createElement('canvas');
      c1.width = midW; c1.height = midH;
      const cx1 = c1.getContext('2d')!;
      cx1.imageSmoothingEnabled = true;
      cx1.imageSmoothingQuality = 'high';
      cx1.drawImage(img, 0, 0, midW, midH);

      // Pass 2 — final scale to 2048px
      const c2 = document.createElement('canvas');
      c2.width = TARGET_W; c2.height = finalH;
      const cx2 = c2.getContext('2d')!;
      cx2.imageSmoothingEnabled = true;
      cx2.imageSmoothingQuality = 'high';
      cx2.drawImage(c1, 0, 0, TARGET_W, finalH);

      URL.revokeObjectURL(url);
      c2.toBlob(
        result => resolve(result || blob),
        'image/jpeg',
        0.97  // 97% quality — visually lossless at 2K
      );
    };
    img.onerror = () => { URL.revokeObjectURL(url); resolve(blob); };
    img.src = url;
  });

// ── Nano Banana Pro Pass A — export so App.tsx can run it ONCE per scene ────────
// Both inframe + outframe share the same character identity — no need to run twice.
// Call before generateCharacterFrame, pass result as preComputedCharIntel.
export const extractCharacterIntelligence = async (
  targetCharacterImages: File[]
): Promise<any> => {
  if (targetCharacterImages.length === 0) return null;
  const charBase64s = await Promise.all(targetCharacterImages.map(fileToBase64));
  const charCount   = charBase64s.length;
  const ai          = getAI();
  const parts: any[] = [
    ...charBase64s.map((b64, i) => ({
      inlineData: { data: b64, mimeType: targetCharacterImages[i].type || 'image/jpeg' }
    })),
    { text: `You are a forensic identity analyst. These ${charCount} photo${charCount > 1 ? 's show' : ' shows'} THE CHARACTER — the same person who must appear in the final generated image with 100% forensic identity fidelity. Cross-reference ALL photos simultaneously. Be hyper-specific. Return ONLY valid JSON, no markdown:
{
  "face_geometry": {
    "overall_shape": "exact face shape and what makes it specific",
    "jaw": "exact width, terminus (sharp/squared/rounded/soft), jawline angle, definition strength",
    "brow_ridge": "prominence above orbital rim, shadow over eye, flat/projecting/moderate",
    "cheekbones": "height, projection, angle, where specular peak falls",
    "orbital_depth": "deep-set/medium/shallow — shadow depth in orbital cavity",
    "nose": "bridge width, tip geometry, length, specific character",
    "philtrum": "length, groove definition",
    "chin": "exact shape, projection, size relative to face",
    "asymmetries": "every left-right difference — brow heights, nostril, lip corners"
  },
  "skin": {
    "tone_precise": "exact undertone + luminosity + zone variations",
    "texture": "pore visibility, smoothness quality, roughness zones",
    "sss_quality": "how light scatters at ears, nasal tip, nasolabial folds",
    "marks": "every mole, scar, mark — exact location and description",
    "life_character": "lines, creases, lived-in characteristics"
  },
  "eyes": {
    "iris_color_precise": "exact color — specific warm-amber/chestnut/slate/hazel quality; color rings",
    "iris_pattern": "fibrous radial structure — dense/open; color zones",
    "limbal_ring": "thickness and darkness — prominent/moderate/subtle",
    "spacing": "interpupillary distance — close/medium/wide",
    "shape": "almond/round/hooded/deep-set; lid crease; lower lid fullness",
    "brow_precise": "arch shape, density, color, natural growth direction, gaps"
  },
  "hair": {
    "color_precise": "exact shade, warm/cool undertone, how it reflects light",
    "texture": "straight/wavy/curly + coarse/fine/medium",
    "density": "thick/medium/thin + hairline character",
    "cut_and_style": "shape, length, how worn",
    "light_behavior": "where specular vs. deep/absorbing"
  },
  "distinguishing_features": ["every unique asymmetry, mark, scar, geometric feature — MUST appear unchanged"],
  "wardrobe": "garment type, exact color, fabric quality, fit, collar, what it signals",
  "environment": {
    "background": "exact colors, elements, tonal quality, depth",
    "bokeh": "how background softens — gradual/abrupt; disc character",
    "atmosphere": "environmental quality, type of space"
  },
  "lighting": {
    "key_direction": "which side, approximate angle",
    "shadow_depth": "ratio impression — shallow/moderate/deep/dramatic",
    "color_temperature": "warm/neutral/cool + Kelvin estimate",
    "rim_light": "any separation light on hair or shoulders"
  },
  "presence_quality": "one sentence: what makes this person compelling in 0.3 seconds"
}` }
  ];
  try {
    const resp = await ai.models.generateContent({
      model: MODEL_TEXT_ELITE,
      contents: [{ role: 'user', parts }],
      config: { thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM } },
    });
    const raw = (resp as any).text ?? (resp as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '{}';
    return safeJsonParse<any>(raw, 'CharacterIntelligence');
  } catch {
    return null;
  }
};

export const generateCharacterFrame = async (
  referenceFrame:        Blob,
  targetCharacterImages: File[],
  role:                  string,
  emotion:               string,
  momentBefore?:         string,
  expressionDirection?:  string,
  physicalSignature?:    string,
  energyLevel?:          number,
  isInframe?:            boolean,
  preComputedCharIntel?: any       // From extractCharacterIntelligence() — skips Pass A
): Promise<{ blob: Blob; enhanced: boolean }> => {

  if (targetCharacterImages.length === 0) return { blob: referenceFrame, enhanced: false };

  // Use ALL character photos — every image strengthens the identity model
  const charBase64s = await Promise.all(
    targetCharacterImages.map(img => fileToBase64(img))
  );
  const refBase64 = await fileToBase64(referenceFrame);

  const charCount = charBase64s.length;

  // ── Nano Banana Pro Agent — 3-Pass Character Swap ────────────────────────────
  // Pass A: Character Intelligence  — text model extracts forensic JSON portrait
  // Pass B: Pose Intelligence        — text model extracts exact pose geometry JSON
  // Pass C: Elite Image Synthesis   — image model gets images + both descriptions
  //
  // Dual-channel anchoring: the image model receives BOTH the actual photos AND
  // precise text descriptions of what those photos contain. This prevents identity
  // averaging and feature blending — the model has independent visual + linguistic
  // anchors pointing at the same truth.
  const ai = getAI();

  // ── Passes A & B run in parallel ─────────────────────────────────────────────
  const [charResult, poseResult] = await Promise.allSettled([

    // ═══ PASS A: CHARACTER INTELLIGENCE (skip if pre-computed) ═══════════════
    // If caller already ran extractCharacterIntelligence(), reuse that result.
    // Otherwise run the full forensic analysis here.
    preComputedCharIntel != null
      ? Promise.resolve(preComputedCharIntel)
      : (async (): Promise<any> => {
      const charParts: any[] = [
        ...charBase64s.map((b64, i) => ({
          inlineData: { data: b64.includes(',') ? b64.split(',')[1] : b64, mimeType: targetCharacterImages[i].type || 'image/jpeg' }
        })),
        { text: `You are a forensic identity analyst and elite cinematography expert. These ${charCount} photo${charCount > 1 ? 's show' : ' shows'} THE CHARACTER — the same person who must appear in the final generated image with 100% forensic identity fidelity. Your analysis feeds directly to an image generation model as the primary identity anchor. Cross-reference ALL ${charCount} photo${charCount > 1 ? 's' : ''} simultaneously. Be hyper-specific — not category labels but exact observed details. Return ONLY valid JSON, no markdown:
{
  "face_geometry": {
    "overall_shape": "exact face shape and what makes it specific to this person",
    "jaw": "exact width, terminus geometry (sharp/squared/rounded/soft), jawline angle ear-to-chin, definition strength",
    "brow_ridge": "prominence above orbital rim, whether it casts shadow over the eye, flat/projecting/moderate",
    "cheekbones": "height, projection from face plane, angle, where specular peak falls in key light",
    "orbital_depth": "deep-set/medium/shallow — describe shadow depth in the orbital cavity",
    "nose": "bridge width (narrow/medium/wide), tip geometry (rounded/refined/broad/pointed), length, specific character",
    "philtrum": "length, groove definition (sharp/soft)",
    "chin": "exact shape, projection from face plane, size relative to face",
    "asymmetries": "every left-right difference — which brow is higher, any nostril/lip corner asymmetry, any side that is fuller or more angular"
  },
  "skin": {
    "tone_precise": "exact undertone (warm/neutral/cool) + luminosity level + any zone variations",
    "texture": "pore visibility and character, smoothness quality, any roughness zones",
    "sss_quality": "how light scatters on this specific skin at ears, nasal tip, nasolabial folds — describe the warm glow quality",
    "marks": "every mole, scar, mark, shadow pattern, characteristic — exact location and description",
    "life_character": "lines, creases, lived-in characteristics that belong to this specific face"
  },
  "eyes": {
    "iris_color_precise": "exact color — not brown/blue but the specific warm-amber/chestnut/slate/hazel quality; color rings or variation zones",
    "iris_pattern": "fibrous radial structure character — dense/open; any distinctive color zones near pupil vs. limbal ring",
    "limbal_ring": "thickness and darkness — prominent/moderate/subtle",
    "spacing": "interpupillary distance relative to face width — close/medium/wide",
    "shape": "almond/round/hooded/deep-set; upper lid crease depth; lower lid fullness",
    "brow_precise": "exact arch shape (high/low/flat/angled), density (thick/medium/thin), color, natural growth direction, any gaps"
  },
  "hair": {
    "color_precise": "exact shade with warm/cool undertone AND how it absorbs vs. reflects — where highlights fall",
    "texture": "straight/wavy/curly + coarse/fine/medium",
    "density": "thick/medium/thin + hairline character",
    "cut_and_style": "specific shape, length, how worn",
    "light_behavior": "how it catches the key light, where specular vs. deep/absorbing"
  },
  "distinguishing_features": ["every unique asymmetry, mark, scar, geometric feature — these MUST appear in the output unchanged"],
  "wardrobe": "full description: garment type, exact color under the photo light, visible fabric quality, fit, collar/lapel character, what it signals",
  "environment": {
    "background": "exact description — colors, visible elements, tonal quality, depth",
    "bokeh": "how the background softens — gradual/abrupt; bokeh disc character (irregular/smooth)",
    "atmosphere": "overall environmental quality and type of space"
  },
  "lighting": {
    "key_direction": "which side the key light hits, approximate angle",
    "shadow_depth": "key-to-fill ratio impression — shallow/moderate/deep/dramatic",
    "color_temperature": "warm/neutral/cool + Kelvin estimate",
    "rim_light": "any separation light visible on hair or shoulder edge"
  },
  "presence_quality": "one sentence: the specific quality that makes this person compelling in 0.3 seconds"
}` }
      ];
      const resp = await ai.models.generateContent({
        model: MODEL_TEXT_ELITE,
        contents: [{ role: 'user', parts: charParts }],
        config: { thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM } },
      });
      const raw = (resp as any).text ?? (resp as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '{}';
      return safeJsonParse<any>(raw, 'CharacterIntelligence');
    })(),

    // ═══ PASS B: POSE INTELLIGENCE ════════════════════════════════════════════
    // Dedicated vision model studies the reference frame → precise pose geometry.
    // This is the ONLY data taken from the reference frame — pure spatial geometry.
    (async (): Promise<any> => {
      const poseParts: any[] = [
        { inlineData: { data: refBase64.includes(',') ? refBase64.split(',')[1] : refBase64, mimeType: 'image/jpeg' } },
        { text: `You are a cinematography analyst. Analyze this reference frame and extract the exact pose geometry. A DIFFERENT PERSON will be placed into this exact spatial arrangement — only abstract geometry matters. The person currently visible is irrelevant. Return ONLY valid JSON, no markdown:
{
  "head_rotation": "degrees turned from camera axis — fully facing (0°) / slight (~15°) / three-quarter (~35°) / near-profile (~60°) / profile (90°); which direction",
  "head_tilt": "lateral cant — left/right/neutral; approximate degree",
  "chin_elevation": "chin up / neutral / slightly down relative to horizon",
  "shoulder_axis": "squared to camera / angled left / angled right / three-quarter; which shoulder is forward",
  "vertical_crop": "tight head crop / head + neck / MCU with chest / medium shot; describe head room",
  "lateral_position": "centered / left of center / right of center; negative space distribution",
  "forward_lean": "toward camera / neutral / away; amount",
  "eye_line": "where eyes fall — upper third / upper-middle / midframe",
  "body_language_essence": "one sentence: the spatial quality of this pose and what it communicates",
  "background_zone": "what is behind and how much negative space on each side"
}` }
      ];
      const resp = await ai.models.generateContent({
        model: MODEL_TEXT_ELITE,
        contents: [{ role: 'user', parts: poseParts }],
        config: { thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM } },
      });
      const raw = (resp as any).text ?? (resp as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '{}';
      return safeJsonParse<any>(raw, 'PoseIntelligence');
    })(),
  ]);

  // ── Extract with fallbacks ────────────────────────────────────────────────────
  const charIntel = charResult.status === 'fulfilled' ? charResult.value : null;
  const poseIntel = poseResult.status === 'fulfilled' ? poseResult.value : null;

  // ── Build linguistic anchors from Pass A + B intelligence ─────────────────────
  const charLines: string[] = [];
  if (charIntel) {
    const fg = charIntel.face_geometry;
    if (fg) charLines.push(`FACE GEOMETRY: ${[fg.overall_shape, fg.jaw, fg.cheekbones, fg.orbital_depth, fg.brow_ridge, fg.nose, fg.chin, fg.asymmetries].filter(Boolean).join(' | ')}`);
    const sk = charIntel.skin;
    if (sk) charLines.push(`SKIN: ${[sk.tone_precise, sk.texture, sk.sss_quality, sk.marks, sk.life_character].filter(Boolean).join(' | ')}`);
    const ey = charIntel.eyes;
    if (ey) charLines.push(`EYES: ${[ey.iris_color_precise, ey.iris_pattern, ey.limbal_ring, ey.spacing, ey.shape, ey.brow_precise].filter(Boolean).join(' | ')}`);
    const hr = charIntel.hair;
    if (hr) charLines.push(`HAIR: ${[hr.color_precise, hr.texture, hr.density, hr.cut_and_style, hr.light_behavior].filter(Boolean).join(' | ')}`);
    if (charIntel.distinguishing_features?.length) charLines.push(`DISTINGUISHING FEATURES (MUST APPEAR UNCHANGED IN OUTPUT): ${(charIntel.distinguishing_features as string[]).join(' | ')}`);
    if (charIntel.wardrobe) charLines.push(`WARDROBE: ${charIntel.wardrobe}`);
    const env = charIntel.environment;
    if (env) charLines.push(`ENVIRONMENT: ${[env.background, env.bokeh, env.atmosphere].filter(Boolean).join(' | ')}`);
    const lt = charIntel.lighting;
    if (lt) charLines.push(`LIGHTING ON CHARACTER: ${[lt.key_direction, lt.shadow_depth, lt.color_temperature, lt.rim_light].filter(Boolean).join(' | ')}`);
    if (charIntel.presence_quality) charLines.push(`PRESENCE: ${charIntel.presence_quality}`);
  }
  const poseLines: string[] = [];
  if (poseIntel) {
    if (poseIntel.head_rotation)         poseLines.push(`HEAD ROTATION: ${poseIntel.head_rotation}`);
    if (poseIntel.head_tilt)             poseLines.push(`HEAD TILT: ${poseIntel.head_tilt}`);
    if (poseIntel.chin_elevation)        poseLines.push(`CHIN: ${poseIntel.chin_elevation}`);
    if (poseIntel.shoulder_axis)         poseLines.push(`SHOULDERS: ${poseIntel.shoulder_axis}`);
    if (poseIntel.vertical_crop)         poseLines.push(`CROP: ${poseIntel.vertical_crop}`);
    if (poseIntel.lateral_position)      poseLines.push(`POSITION IN FRAME: ${poseIntel.lateral_position}`);
    if (poseIntel.forward_lean)          poseLines.push(`LEAN: ${poseIntel.forward_lean}`);
    if (poseIntel.eye_line)              poseLines.push(`EYELINE: ${poseIntel.eye_line}`);
    if (poseIntel.body_language_essence) poseLines.push(`POSE ESSENCE: ${poseIntel.body_language_essence}`);
    if (poseIntel.background_zone)       poseLines.push(`BACKGROUND ZONE: ${poseIntel.background_zone}`);
  }

  const charDescriptionBlock = charLines.length > 0
    ? `EXTRACTED CHARACTER PORTRAIT — forensic analysis of ${charCount} identity photo${charCount > 1 ? 's' : ''} by dedicated vision model:\n${charLines.join('\n')}`
    : `CHARACTER: The person in the ${charCount} attached identity photo${charCount > 1 ? 's' : ''} — use as absolute identity reference.`;

  const poseDescriptionBlock = poseLines.length > 0
    ? `EXTRACTED POSE GEOMETRY — from dedicated reference frame analysis:\n${poseLines.join('\n')}`
    : `POSE: Match the spatial arrangement visible in the attached reference frame.`;

  // ── Pass C: Build synthesis prompt (agent intelligence + rendering directives) ─
  const isIn = isInframe !== false;
  const energyLabel = (energyLevel ?? 7) >= 8
    ? 'HIGH AUTHORITY DRIVE — forward energy, full presence, conviction already loaded before the first word'
    : (energyLevel ?? 7) >= 6
    ? 'MID AUTHORITY — warm, deliberate, settled — the quality of someone with complete command of what they are about to give'
    : 'INTIMATE REGISTER — low, warm, close — the private transmission of someone choosing to share something true';

  const frameTypeBlock = isIn
    ? `═══════════════════════════════════════════════════════════════
FRAME TYPE: OPENING (INFRAME) — PRE-SPEECH LOADED STATE
═══════════════════════════════════════════════════════════════
This frame captures the exact moment 0.3–0.5 seconds BEFORE the first word arrives. The performance has begun internally — the voice has not yet activated. This is not neutral waiting. It is inhabited readiness.

THE INTERNAL STATE: THE CHARACTER has just drawn a complete breath, the thought is fully loaded, the target viewer has been located behind the lens, and the first word is milliseconds away. Everything is ready. The face knows exactly what it is about to give.

PHYSICAL REALITY OF THIS MOMENT:
· Lips: at natural biological rest — softly separated, neither pressed closed nor open for speech; the ease of lips with no agenda yet
· Jaw: settled at its natural resting position — the specific fleshy weight of a jaw not yet engaged in articulation; soft and floating
· Chest: the subtle fullness of a completed inbreath — sternum fractionally elevated, the quiet loading of breath support
· Eyes: making genuine contact with the lens — not performing eye contact but actually looking at the specific person behind the glass; carrying the full weight of the thought about to be given; natural moisture, a recent blink just completed, organically alive
· The face as a whole: the quality of absolute quiet readiness — "I have everything. The first word is about to arrive."
· Brow: completely smooth — not a trace of effort or anticipation; the calm of someone for whom this is easy
· The expression reads: SETTLED AUTHORITY ABOUT TO SPEAK — not anticipating, not telegraphing; simply being, loaded${momentBefore ? `\n\nDIRECTOR'S MOMENT-BEFORE DIRECTION:\n${momentBefore.substring(0, 600)}` : ''}`
    : `═══════════════════════════════════════════════════════════════
FRAME TYPE: CLOSING (OUTFRAME) — POST-SPEECH INHABITED RESIDUE
═══════════════════════════════════════════════════════════════
This frame captures the moment immediately after the final word has completed and the voice has settled. The performance is not over — the face carries the emotional residue of what was just given. This is not neutral. It is the specific inhabited quality of someone who meant every word.

THE INTERNAL STATE: THE CHARACTER has just placed something of genuine value on the table. The transmission is complete. The body is returning to rest — not resetting to neutral, but settling with the organic weight of authentic completion.

PHYSICAL REALITY OF THIS MOMENT:
· Lips: beginning their return to natural rest — jaw decelerating from its last articulation; the biological loosening after speech
· Chest: releasing on a quiet organic exhale — the natural fall after supported speech
· Eyes: still holding the lens — not searching, not releasing; the warm present quality of someone still with the viewer after giving something real; natural moisture, organically alive
· Expression: the "after" — not neutral, not performing, but carrying the visible weight of having just transmitted something true
· The face says: "That was real. I meant it. I'm still here." — the quality of genuine completion without detachment`;

  const prompt = `
═══════════════════════════════════════════════════════════════
⚠⚠ MANDATORY PROCESSING SEQUENCE — EXECUTE IN EXACT ORDER ⚠⚠
Do NOT begin rendering until all three analysis steps are complete.
═══════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — BUILD THE CHARACTER MODEL (Images 1–${charCount} ONLY)
Cross-reference ALL ${charCount} identity photo${charCount > 1 ? 's' : ''} simultaneously to build the most complete forensic portrait possible.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extract and lock these dimensions of THE CHARACTER's identity:

FACIAL GEOMETRY — the spatial architecture of this specific face:
· Jaw: its exact width, terminus geometry (sharp/soft/squared), the angle of the jawline from ear to chin
· Brow ridge: prominence above the orbital rim — does it cast shadow over the eye? Flat/projecting/moderate?
· Cheekbone plane: angle and height — how does it catch light? Flat against the face or projecting?
· Orbital depth: how deep-set are the eyes? What shadow falls into the orbital cavity?
· Nasal structure: bridge width, tip geometry, length relative to midface
· Philtrum: length and definition of the groove from nose to upper lip
· Chin: exact shape — rounded/squared/pointed; projection from face plane; the way it occupies space

SKIN — this person's specific skin, not a generic version:
· Tone: the precise warm/neutral/cool undertone; the specific luminosity quality
· Texture: pore character, smoothness, any zones of roughness or sheen
· SSS profile: how light penetrates and scatters on THIS specific skin (more evident on ears, nasal tip)
· Any marks, variations, or characteristics that are part of this face — these make it real

EYES — the most identity-critical feature:
· Iris color: precise — not "brown" but the exact warm/cool amber/chestnut/slate quality and how it catches light
· Iris pattern: radial structure character, any distinct coloration zones
· Limbal ring: presence, thickness, gradient
· Eye spacing: exact interpupillary relationship
· Lid shape: upper lid crease depth, natural aperture, lower lid fullness
· Brow: shape, density, the natural arch character — these are identity markers

HAIR — every strand of identity:
· Color: exact — the specific chestnut/slate/warm-espresso shade AND how it reflects/absorbs under the source light
· Texture: straight/wavy/coarse/fine — the way it falls and holds shape
· Density: how full is it; how does it behave at the hairline
· Cut and style: the specific shape and length that defines this person

DISTINGUISHING FEATURES — the imperfections that make this face forensically unique:
· Every asymmetry (left vs. right brow height, nostril, lip corner)
· Any marks, scars, shadows, or characteristics that belong to this face
· The specific micro-geometry that no other face shares
· These must survive into the output — they are not flaws, they are identity

WARDROBE — the full visual character signal:
· Fabric type and weight as visible in the photos
· Color exactly as it reads under the photo's light
· Collar and cut
· What it signals to someone who reads clothing fluently

ENVIRONMENT — the setting THE CHARACTER inhabits:
· Background: exact tones, depth, spatial geometry, any visible elements
· Bokeh quality: how the background softens; the organic character of the out-of-focus zone
· Lighting character: key light direction and quality as visible on THE CHARACTER's face; the specific way their skin renders under this light; color temperature
· Atmosphere: the full environmental quality — what kind of space this is, what it communicates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — EXTRACT POSE GEOMETRY ONLY (Image ${charCount + 1} ONLY)
Everything from this image except the five geometric parameters below is SURGICALLY REMOVED from your working memory.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Extract ONLY these five abstract spatial parameters:
1. HEAD ROTATION: how many degrees is the head turned toward or away from the camera axis? (0° = fully facing camera; 30° = three-quarter; 90° = full profile)
2. HEAD TILT: lateral cant — which direction and how much (left lean / right lean / neutral)
3. SHOULDER AXIS: the angle of the shoulder line relative to the camera plane — squared / angled / three-quarter
4. VERTICAL CROP: where in the frame does the head sit — high / mid / with visible chest / tight crop
5. FORWARD LEAN: is the torso leaning toward camera, away, or neutral — and by how much

THESE FIVE PARAMETERS ARE ALL THAT SURVIVES FROM IMAGE ${charCount + 1}.
The person's face, identity, skin, hair, clothing, expression, background, lighting, color — ALL DISCARDED.
Zero visual DNA from the pose reference person enters the output. Zero.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — SYNTHESIS: THE CHARACTER in THE POSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Render THE CHARACTER exactly as modeled in Step 1 — same face, same identity, same wardrobe, same environment, same lighting — but repositioned to the five geometric parameters extracted in Step 2. Calibrate expression for "${emotion}" at energy level ${energyLevel ?? 7}/10 for a ${role} scene.

IDENTITY PRIORITY LAW: If there is any conflict between pose precision and identity fidelity, IDENTITY WINS EVERY TIME. A slightly different pose angle with perfect identity is a success. The same pose with any identity drift is a complete failure.

═══════════════════════════════════════════════════════════════
ENVIRONMENT — FROM CHARACTER PHOTOS, ELEVATED TO CINEMA GRADE
═══════════════════════════════════════════════════════════════
The background, atmosphere, bokeh, depth, and spatial geometry come EXCLUSIVELY from THE CHARACTER's identity photos. The pose reference contributes ZERO to the environment.

SPATIAL FIDELITY: Preserve every structural element of the environment:
· Background content and depth relationship — exactly as in the identity photos
· Bokeh character: the specific organic softness of out-of-focus elements; bokeh discs with slight natural irregularity (not CGI-perfect circles)
· Atmospheric quality: the specific sense of space and dimension
· Subject-to-background distance relationship

LIGHTING — EVALUATE AND EXECUTE:
If the character photos show cinematic quality (distinct key light, facial modeling, subject-to-background separation, no flat overhead cast) → PRESERVE EXACTLY.
If the lighting is flat, overhead, mixed-temperature, or lacks facial dimensionality → UPGRADE to elite cinema standard while keeping the background identical:
· KEY: warm amber (5600K) at 45° lateral to the face — creates dimensional cheekbone modeling, draws out bone structure; large parabolic soft modifier for smooth falloff
· FILL: deep ratio (8:1) — significant shadow defines the non-key side; authority and dimension
· RIM: cool blue-silver separation light from behind the opposite shoulder — creates a 3D halo that lifts the subject off the background; prevents subject merging into the space
· BACKGROUND LIGHT: cooler teal pushed into the background — Orange & Teal complementary architecture; warm skin tone against cool space; the most psychologically compelling visual hierarchy for human facial content
The upgrade adds cinema dimensionality — it does not change the background content or the environment.

═══════════════════════════════════════════════════════════════
FRAMING — 16:9 CINEMA BROADCAST STANDARD
═══════════════════════════════════════════════════════════════
Format: 2048×1152 px (2K), 16:9 landscape. Maximum quality output.
Framing: Match the pose geometry from Step 2. The face is the dominant visual element.
Lens simulation: 85mm equivalent — produces natural face geometry without distortion; shallow DOF with gradual background falloff; slight compression that reads as authority and expertise.
Composition: Natural broadcast composition — intentional space on the speaking side where applicable. Clean, considered. No awkward crops. No tangent edges.
Focus: Critical plane on the near eye. The eye must be in razor-sharp focus.

${frameTypeBlock}

═══════════════════════════════════════════════════════════════
EXPRESSION ARCHITECTURE — "${emotion}" | ${role} | Energy ${energyLevel ?? 7}/10
═══════════════════════════════════════════════════════════════
ENERGY CALIBRATION: ${energyLabel}
${expressionDirection ? `\nDIRECTOR'S EXPRESSION DIRECTION:\n${expressionDirection.substring(0, 400)}\n` : ''}${physicalSignature ? `PHYSICAL SIGNATURE OF THIS SCENE: ${physicalSignature.substring(0, 300)}\n` : ''}
THE EXPRESSION EMERGES FROM AN INTERNAL STATE, NOT FROM A FACE DECIDING TO SHOW SOMETHING.
The internal state for this scene: "${emotion}" as experienced by someone whose authority is beyond question — not performed, simply present.

THE BIOLOGICAL BASELINE — non-negotiable foundation beneath every expression:
THE PRESENTER's face is the face of mastery at rest. Genuine confidence is a biological state, not a performed one. The foundation:
· Brow: completely smooth and unhurried — every micro-tension here reads as uncertainty in 0.2 seconds; zero tension anywhere across the brow surface; the soft, fleshy, unlocked forehead of someone whose mind has already settled
· Inner brow territory: wide open and spacious — the neurological signature of certainty; anxiety and effort physically compress this space; its openness reads as "I have already decided" in 0.1 seconds
· Chin: fleshy, heavy, and completely still — the single most legible confidence signal on the human face; not controlled stillness but genuine biological heaviness with no reason to move; any dimple, bunch, or tension here reads as suppressed doubt
· Jaw: soft and floating at natural biological rest — slightly asymmetric as real jaw-at-rest always is; the specific fleshy weight of a jaw genuinely not braced for anything; between words it returns to this state instantly
· Lips: their exact natural rest position — slightly asymmetric, softly separated, neither pressed nor held open; carrying no expression between intentions; simply present
· Eyes: the natural focused aperture of someone looking at something they fully understand — not wide, not narrowed; the EXPERT EYE quality of carrying more than they release; genuine warmth in the outer corners only if "${emotion}" includes warmth; natural moisture, spontaneous blink completed, organically alive; directed at the specific location of the camera lens
· Cheeks: ${emotion.includes('warm') || emotion.includes('invitation') || emotion.includes('connect') || role === 'Call to Action' || role === 'Bridge' ? 'genuine Duchenne activation — eye corners and cheeks activate simultaneously and involuntarily; the organic output of actual warmth, not a conscious expression choice; both sides slightly but not symmetrically lifted' : 'at neutral-warm — soft organic rest without active warmth signals; the contained positive regard of an expert who cares but does not perform caring'}

EXPRESSION SEQUENCING — the biology of natural expression:
Eyes respond first. The thought arrives in the eyes a fraction of a second before the lower face moves. Lower face follows. Expression reaches peak. Returns to biological rest. Never instantaneous. Never performed. The difference between genuine expression and AI expression is entirely in this sequencing.

NATURAL ASYMMETRY — the biological signature of authentic feeling:
Every genuine human expression is organically asymmetric — dominant hemisphere leads fractionally, left and right sides never reach peak simultaneously, the eye corners activate at slightly different moments. Perfect bilateral symmetry is the primary AI expression tell. Build in the natural organic asymmetry of a real face carrying a real internal state.

BLINK STATE: Caught between completed blinks or immediately after a completed blink — not mid-blink, not artificially wide-open. The natural organic blink state of a living face.

═══════════════════════════════════════════════════════════════
CINEMA HYPER-REALISM PHYSICS — MAXIMUM FIDELITY
═══════════════════════════════════════════════════════════════

SKIN PHYSICS — six-layer model:
· SUBSURFACE SCATTERING: ears carry warm pinkish-red translucency as light penetrates cartilage; nasal tip has similar warm glow; nasolabial folds show amber undertone where scattered light re-emerges; cheekbones at the steepest key-light angle show apricot specular; these SSS zones are the primary realism signal
· PORE ARCHITECTURE: in key-light zones (forehead, nose bridge, cheek planes), individual pore rims catch micro-shadows — genuine topographic relief, not noise-mapped texture; actual surface variation that reads as real skin at 2K
· FRESNEL REFLECTANCE: at glancing angles (jaw edge, lateral cheekbone, orbital rim, ear rim, temple), skin becomes more specular — wider, brighter reflections; at normally-incident zones (forehead center, nose bridge), more diffuse; this angle-dependent reflectance is what makes the face genuinely three-dimensional
· SEBUM DIFFERENTIAL: the T-zone (forehead center, nose, chin) carries marginally higher specular return than the lateral matte-diffuse cheeks — a natural skin quality visible in real photography
· VELLUS HAIR: at the cheekbone edge, jaw perimeter, and hairline, fine vellus hair is visible as a barely-there luminous bloom in direct key light — not stubble, not texture; a translucent organic halo
· MICRO-DEFORMATION: skin over the jaw and chin region shows organic stretch and compress appropriate to the expression — real flesh on bone moves; the tissue is not rendered static

EYE PHYSICS — the most scrutinized element at 2K:
· IRIS DEPTH: radial fibrous structure with crypts and ridges clearly visible; not a flat color disc; color transitions from deep rich at the pupil border to mid-tone to slightly lighter near the limbal ring; the three-dimensional quality of a real iris
· LIMBAL RING: dark graduated band at the iris-sclera boundary; clearly present; 1–2mm; graduated (not a hard line); the limbal ring is a biological authenticity signal — its absence or sharpness reads as AI
· TEAR FILM: a narrow bright specular line along the lower lid margin — the optical signature of a moist, living eye; this single element contributes enormously to perceived aliveness
· SCLERAL CHARACTER: warm cream undertone, not clinically white; faint capillary traces at medial and lateral canthi — blood vessels in the white of the eye; these signal biological life
· CATCHLIGHTS: two lights — primary warm (upper-third of iris, larger, warm amber cast from key light) + secondary cool (smaller, cooler, opposite side from fill); their precise positions locked to the established lighting; both clearly present
· PUPIL: at natural dilation for this light environment; soft organic boundary; not perfectly circular — very slight natural irregularity

HAIR PHYSICS — individual strand precision:
· Individual strands visible and distinct at hairline, part line, and around the ears — not a silhouette mass; actual strand-by-strand rendering where individual hairs can be traced
· Crown specular highlight: positioned precisely to the key light angle; warm toned from the amber key; shifts with head angle
· Hair-to-skin transition at the forehead: graduated, with baby hairs and varying strand density — the organic interface between hair and scalp
· Color precision: not a category but the exact shade — with its specific warm/cool undertone — AND how this particular hair absorbs and reflects under this specific light; where it catches specular highlights; where it goes deep

FABRIC PHYSICS — textile at 2K:
· Weave character: actual thread structure visible at close framing; the specific textile pattern of this fabric; not texture noise but genuine textile rendering
· Gravity fall: fabric hangs with the specific weight of its material; drape at the shoulder joint; collar falls naturally
· Micro-wrinkle: at the shoulder joint, at any flex points — real cloth wrinkles from body movement; peaks more specular, valleys deeper
· Color under this light: fabric reads under the color temperature of the scene; not generic

LENS PHYSICS — the optics of authenticity:
· Depth of field: critical plane on the near eye; the tip of the nose begins its subtle softening; background softens progressively beginning approximately at the ear plane; the separation is gradual, not hard
· Bokeh quality: out-of-focus elements render as organic, slightly irregular discs — not CGI-perfect circles; real glass has slight asymmetry in bokeh character
· Micro-contrast: the specific local contrast character of a real lens — slightly harder at points of critical focus, softer at the DOF boundaries; the visual equivalent of film grain at the micro level
· No chromatic aberration or optical artifacts — this is a premium lens; clean rendering at all edges

DEPTH AND DIMENSION:
· Subject exists in three-dimensional space, not as a figure placed in front of a background
· The relationship between subject and background is spatial — there is genuine air between them
· The subject's shoulders, neck, and ears exist at different depth planes
· Rim lighting creates physical separation — the subject has three-dimensional presence

═══════════════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS — these destroy cinema realism instantly
═══════════════════════════════════════════════════════════════
· NO AI skin smoothing, plastic luminosity, or beautification — real skin has zones, texture, variation
· NO skin uniformity — perfect uniform skin reads as CGI in 0.1 seconds
· NO artificially white sclera — clinical white eyes are a primary AI tell; warm cream undertone only
· NO CGI sheen or plastic reflectance on any surface
· NO hard silhouette edges at hair perimeter — hair dissolves into air at the edges
· NO perfect bilateral symmetry in any expression — natural asymmetry is the biological signature of genuine feeling
· NO composite artifacts — the subject must exist seamlessly within their environment
· NO performed expression — if the biological baseline (smooth brow, still chin, floating jaw, natural lips) is absent, the expression reads as AI in 0.1 seconds
· NO identity blending — not a single feature from Image ${charCount + 1}'s person enters the output
· NO generic version of THE CHARACTER — the output must be THIS specific person from the identity photos, forensically verifiable

═══════════════════════════════════════════════════════════════
FINAL QUALITY VERIFICATION — before rendering output
═══════════════════════════════════════════════════════════════
The output passes if and only if ALL of the following are true:
✓ The face in the output is forensically verifiable as THE CHARACTER from Images 1–${charCount}
✓ The pose geometry matches the five parameters extracted from Image ${charCount + 1}
✓ The background and environment match the character's identity photo setting
✓ The expression communicates "${emotion}" from an internal state, not a performed pose
✓ The biological baseline (smooth brow, still chin, floating jaw, natural lips) is intact
✓ Both catchlights are present and positioned correctly
✓ Tear film specular line is visible along the lower lid margin
✓ The face reads as genuinely alive — not posed, not frozen, not AI

OUTPUT: One single photograph. 2K resolution (2048×1152). Photorealistic. Cinema grade. Nothing else.
`;

  // ── Build final synthesis prompt: agent intelligence header + rendering directives
  // The image model sees the extracted descriptions FIRST (highest attention weight),
  // then the full rendering directive. Both images AND matching text anchors = maximum
  // identity fidelity.
  const synthPrompt = `═══════════════════════════════════════════════════════════════
⚠ CHARACTER IDENTITY SWAP — AGENT SYNTHESIS (Pass C)
═══════════════════════════════════════════════════════════════
TASK: Produce a single 2K photorealistic photograph of THE CHARACTER (identity photos attached) placed in the pose geometry of the reference frame. Expression: "${emotion}" for ${role} at energy ${energyLevel ?? 7}/10.

IMAGE ROLES:
· Images 1–${charCount}: THE CHARACTER — source of ALL identity, face, skin, hair, wardrobe, environment, and atmosphere
· Image ${charCount + 1}: POSE GEOMETRY ONLY — spatial parameters extracted; the person in this image is COMPLETELY REPLACED

IDENTITY PRIORITY LAW: Identity fidelity > Pose precision > everything else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DUAL LINGUISTIC ANCHORS — extracted by dedicated vision models
Match BOTH the attached images AND these text descriptions exactly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

${charDescriptionBlock}

${poseDescriptionBlock}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RENDERING DIRECTIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${prompt}`;

  // ── Pass C: parts — character photos first, reference frame second, text last ──
  const synthParts: any[] = [
    ...charBase64s.map((b64, i) => ({
      inlineData: { data: b64.includes(',') ? b64.split(',')[1] : b64, mimeType: targetCharacterImages[i].type || 'image/jpeg' }
    })),
    { inlineData: { data: refBase64.includes(',') ? refBase64.split(',')[1] : refBase64, mimeType: 'image/jpeg' } },
    { text: synthPrompt },
  ];

  try {
    const response = await ai.models.generateContent({
      model: MODEL_IMAGE_GEN,
      contents: [{ role: 'user', parts: synthParts }],
      config: {
        responseModalities: ['TEXT', 'IMAGE'],
        imageConfig: { imageSize: '2K', aspectRatio: '16:9' },
      },
    });
    const rawBlob = extractImageFromResponse(response);
    if (rawBlob) {
      const blob2k = await upscaleTo2K(rawBlob);
      return { blob: blob2k, enhanced: true };
    }
  } catch (err) {
    console.error('[generateCharacterFrame Pass C] Image synthesis failed:', err);
  }

  // Fallback: return first uploaded character photo upscaled to 2K
  const fallbackB64 = await fileToBase64(targetCharacterImages[0]);
  const fallbackRaw = fallbackB64.includes(',') ? fallbackB64.split(',')[1] : fallbackB64;
  const bytes = atob(fallbackRaw);
  const arr   = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
  const fallbackBlob = new Blob([arr], { type: targetCharacterImages[0].type || 'image/jpeg' });
  const fallback2k   = await upscaleTo2K(fallbackBlob);
  return { blob: fallback2k, enhanced: false };
};

const extractImageFromResponse = (response: any): Blob | null => {
  const parts = response.candidates?.[0]?.content?.parts || [];
  for (const part of parts) {
    if (part.inlineData) {
      const bytes = atob(part.inlineData.data);
      const arr   = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
      return new Blob([arr], { type: part.inlineData.mimeType || 'image/jpeg' });
    }
  }
  // Log to help diagnose when the model responds without an image
  const textParts = parts.filter((p: any) => p.text).map((p: any) => p.text).join(' ');
  console.warn('[extractImageFromResponse] No image in response. Text parts:', textParts || '(none)', '| Full response:', JSON.stringify(response).slice(0, 500));
  return null;
};

// ============================================================
// FUNCTION 4 — Engineer Scene Prompt (Oscar-Level VEO 3.1)
// ============================================================
export const engineerScenePrompt = async (
  scene:                 ScriptScene,
  referenceAnalysis:     ReferenceAnalysis,
  inframeImage:          File,
  outframeImage:         File,
  targetCharacterImages: File[],
  completedScenes:       EngineeredScene[],
  allScenes?:            ScriptScene[],
  videoArc?:             {
    narrative_arc:    string;
    total_scenes:     number;
    directing_vision: {
      voice_fingerprint?:    string;
      energy_arc_map?:       string;
      presentation_persona?: string;
      through_action?:       string;
    };
  },
  phonemicPrecomp?: string,
  learnedPrefs?:   { gear?: number; focal?: string; energy?: number },
  scene1Lock?:     string
): Promise<string> => {

  const inframeB64  = await fileToBase64(inframeImage);
  const outframeB64 = await fileToBase64(outframeImage);
  const charBase64s = await Promise.all(
    targetCharacterImages.slice(0, 5).map(img => fileToBase64(img))
  );

  const charCount      = charBase64s.length;
  const isAnchorScene  = completedScenes.length === 0;

  // ── Short-scene optimization ────────────────────────────────
  // 7-8s is VEO's sweet spot: enough canvas for natural pre-speech + speech + settle.
  // Scenes ≤5.5s need an explicit articulatory warm-up (face already in motion before
  // words begin) — this mirrors the "gibberish pre-roll" technique that produces
  // natural speech onset in short clips.
  const isShortScene  = scene.duration_seconds <= 5.5;
  const speechSeconds = isShortScene
    ? Math.max(scene.duration_seconds - 1.3, 1.5)   // 0.8s warm-up + 0.5s settle
    : scene.duration_seconds;
  const speechWpm = Math.round(
    ((scene.word_count ?? scene.script_text.trim().split(/\s+/).length) / speechSeconds) * 60
  );

  const charImageLabel = charCount === 1
    ? 'Image 3 is the TARGET CHARACTER — the person who must appear in this video.'
    : `Images 3 through ${charCount + 2} are the TARGET CHARACTER — ${charCount} photos of the same person for maximum identity fidelity.`;

  // ── Persistent character ID — same label used in EVERY scene for VEO consistency ──
  const charName = referenceAnalysis.character?.appearance
    ? referenceAnalysis.character.appearance.split(/[,.(]/)[0].trim()
    : 'THE PRESENTER';
  const charIdLabel = [
    charName.toUpperCase(),
    referenceAnalysis.character?.gender || '',
    referenceAnalysis.character?.age_range || '',
    referenceAnalysis.character?.hair || '',
    referenceAnalysis.character?.skin_tone || '',
  ].filter(Boolean).join(' | ');
  // Short token VEO uses to recognise the character across every scene
  const charToken = `[CHARACTER: ${charIdLabel}]`;

  // ── Voice fingerprint ──────────────────────────────────────
  const voiceFingerprint = [
    referenceAnalysis.character?.voice?.texture,
    referenceAnalysis.character?.voice?.energy_baseline,
    referenceAnalysis.character?.voice?.placement,
    referenceAnalysis.character?.voice?.qualities?.join(', '),
  ].filter(Boolean).join(' | ') || 'warm, grounded chest resonance — the private briefing voice of someone who has earned every word they speak';

  // ── Character Voice Identity Label ────────────────────────
  // A locked identity spec VEO uses to maintain the same voice character across every scene
  const charGender       = referenceAnalysis.character?.gender || 'male';
  const charAgeRange     = referenceAnalysis.character?.age_range || '40-55';
  const charAccent       = referenceAnalysis.character?.voice?.accent || 'US General American';
  const charVoicePitch   = referenceAnalysis.character?.voice?.pitch || 'low-mid register';
  const charVoicePace    = referenceAnalysis.character?.voice?.pace_range || '110-150 WPM';
  const charVoiceTexture = referenceAnalysis.character?.voice?.texture || 'rich chest resonance with controlled authority';
  const characterVoiceLabel = `CHARACTER VOICE IDENTITY (LOCKED — NEVER VARIES ACROSS SCENES):
▸ Identity: ${charGender}, ${charAgeRange}, ${charAccent} accent
▸ Register: ${charVoicePitch} — ${charVoiceTexture}
▸ Baseline pace: ${charVoicePace}
▸ Placement: ${referenceAnalysis.character?.voice?.placement || 'forward chest-to-mask placement, 60% chest / 40% mask resonance'}
▸ Qualities: ${referenceAnalysis.character?.voice?.qualities?.join(', ') || 'authoritative clarity, warm depth, deliberate articulation'}
▸ Energy baseline: ${referenceAnalysis.character?.voice?.energy_baseline || 'controlled intensity — never shouts, never whispers without purpose'}
This is the voice's DNA — its acoustic fingerprint. Every frame of every scene in this video produces exactly this voice. Not approximately. Not inspired by. THIS voice. If the voice in this scene sounds different from this specification in any quality, the scene has failed identity fidelity.`;

  const personaSummary = referenceAnalysis.character?.acting_style?.persona_summary
    || 'peer-level authority — speaks as an equal to sophisticated principals, never performing, always genuine';

  // ── Mouth DNA for lip sync ─────────────────────────────────
  const mouthDna         = referenceAnalysis.character?.mouth_dna;
  const mouthRestPos     = mouthDna?.rest_position        || 'lips at 3-5mm natural separation, jaw relaxed, mentalis easy — the unselfconscious rest of someone mid-thought';
  const articulStyle     = mouthDna?.articulation_style   || 'forward-placed articulation — labials visible, bilabials close fully, sibilants forward and crisp';
  const jawOpenness      = mouthDna?.jaw_openness         || 'moderate jaw travel — 8-10mm on stressed open vowels, 3mm on closed vowels and consonant clusters';
  const preSpeechBehav   = mouthDna?.pre_speech_behavior  || 'a quiet chest expansion (0.3-0.4s), lips parting from rest position before the voice activates — the breath that carries the first word';
  const consonantChar    = mouthDna?.consonant_character  || 'plosives fully released — no glottal substitution; fricatives forward-shaped; final consonants complete and present';
  const breathVis        = mouthDna?.breath_visibility    || 'chest rise visible before phrases; inter-phrase micro-breath at natural punctuation';
  const voicePlacement   = referenceAnalysis.character?.voice?.placement || 'forward-placed — 60% chest resonance, lip activity moderate-to-high';
  const momentBeforeArch = referenceAnalysis.character?.acting_style?.moment_before_archetype
    || 'the body settles before speaking — weight drops through the spine, eyes arrive at the lens, the breath prepares, and the performance begins before the voice does';
  const speechOnsetPhoneme = (scene.acting_blueprint as any).speech_onset_phoneme as string | undefined;
  const firstWord          = scene.script_text.trim().split(/\s+/)[0] || '';

  // ── Inframe/outframe prose descriptions (text fallback for VEO) ─
  const parseFrameTs = (ts: string): number => {
    const p = ts.replace(',', '.').split(':').map(parseFloat);
    return p.length === 3 ? p[0]*3600 + p[1]*60 + p[2]
         : p.length === 2 ? p[0]*60 + p[1]
         : p[0] || 0;
  };
  const buildFrameProse = (timestamp: string, rationale: string): string => {
    const lib = referenceAnalysis.frame_library || [];
    if (!lib.length) return rationale || '';
    const tSec  = parseFrameTs(timestamp);
    const match = lib.reduce((best, f) =>
      Math.abs(parseFrameTs(f.timestamp) - tSec) < Math.abs(parseFrameTs(best.timestamp) - tSec) ? f : best
    , lib[0]);
    return [
      match.body_position && `body: ${match.body_position}`,
      match.expression    && `expression: ${match.expression}`,
      match.mouth_state   && `mouth: ${match.mouth_state}`,
      rationale           && `director note: ${rationale}`,
    ].filter(Boolean).join(' | ');
  };
  const inframeProse  = buildFrameProse(
    scene.recommended_inframe?.timestamp  || '00:00',
    scene.recommended_inframe?.rationale  || ''
  );
  const outframeProse = buildFrameProse(
    scene.recommended_outframe?.timestamp || '00:00',
    scene.recommended_outframe?.rationale || ''
  );

  // ── Continuity context ─────────────────────────────────────
  const continuity = isAnchorScene
    ? `ANCHOR SCENE — Scene 1. Every visual constant you establish here is LOCKED for the entire video. Define each with the specificity that lets subsequent scenes match it frame-perfectly:
— Skin rendering quality: subsurface scattering depth, pore visibility, luminosity character
— Lighting signature: key light direction and temperature, fill depth and ratio, shadow character
— Background: exact tones, depth gradient, bokeh character at this aperture
— Color temperature: measured in Kelvin, as it reads on skin and fabric
— Framing language: camera-to-subject distance, head-to-frame proportions, negative space
— Camera-lens relationship: focal length choice and its specific effect on face geometry`
    : `VISUAL CONTINUITY — zero deviation from the established world.
${scene1Lock ? `\n⚡ SCENE 1 VISUAL CONSTANTS — LOCKED ACROSS ALL SCENES (the attributes VEO drifts on most — match these exactly):\n${scene1Lock}\n` : ''}
LOCKED CONSTANTS (every physical detail frozen from Scene 1):
${completedScenes.slice(-2).map(s => `Scene ${s.scene_number} — "${s.scene_title}":\n${s.veo_prompt.substring(0, 1200)}`).join('\n\n---\n\n')}

Scene transition: ${scene.continuity?.enters_from || 'continues from previous scene energy'} → this scene.
Exits to: ${scene.continuity?.exits_to || 'next scene'}.

EXPRESSION-INHERITANCE PROTOCOL: ${scene.continuity?.expression_inheritance || 'The face carries a residue of the previous scene\'s emotional state for the first 0.3-0.5 seconds — a natural carry-through that signals genuine emotional continuity rather than a scene-by-scene reset. This inherited expression is visible, then organically dissolved by this scene\'s own emotional truth. Not erased instantly — dissolved, as one human feeling transitions into the next.'} The viewer sees the previous scene still alive in the face as this scene begins. This is the signal of continuous performance, not assembled clips.`;

  // ── Extract scene-level blueprint fields ──────────────────
  const masterTA      = videoArc?.directing_vision.through_action || '';
  const sceneTABase   = scene.acting_blueprint.through_action || `To make the viewer feel ${scene.emotional_tone} with the conviction that only genuine authority can create`;
  const throughAction = (masterTA && masterTA.trim() !== sceneTABase.trim())
    ? `${sceneTABase} — advancing master arc: "${masterTA.substring(0, 120)}${masterTA.length > 120 ? '…' : ''}"`
    : sceneTABase;
  const momentBefore     = scene.acting_blueprint.moment_before     || `The thought is fully formed. The breath has been taken. The eyes are already at the lens. From this stillness — the first word arrives.`;
  const lipBlueprint     = scene.acting_blueprint.lip_sync_blueprint;
  const phonemicAnchors  = lipBlueprint?.phonemic_anchors?.join('\n· ') || 'All stressed vowels at full jaw-open width for this character; bilabials fully closed and released; sibilants forward-placed';
  const jawMap           = lipBlueprint?.jaw_travel_map       || 'moderate jaw travel — opens on stressed vowels, closes cleanly between words';
  const lipTension       = lipBlueprint?.lip_tension_notes    || 'clean bilabial releases — full closure, immediate release; labial contacts precise without lingering';
  const breathPoints     = lipBlueprint?.breath_points?.join(' | ') || `one visible breath before the first word (0.3-0.4s chest expansion); natural inter-phrase breath at punctuation`;
  const coArticulation   = lipBlueprint?.co_articulation_notes || 'words flow continuously — co-articulation overlap at word boundaries; no mechanical phoneme-by-phoneme separation';

  // ── Director-level retention fields ───────────────────────
  const retentionTechnique = scene.continuity?.retention_technique
    || `PEER_RECOGNITION — peer-level register and UHNWI-appropriate vocabulary signal in-group membership throughout the ${scene.role} delivery`;
  const emotionalDeposit   = scene.continuity?.emotional_deposit
    || `genuine authority and intellectual value through the specificity and conviction of this ${scene.role} moment`;

  // ── Bible-derived acting directives ───────────────────────
  const vocalGear = (scene.acting_blueprint as any).vocal_gear as number | undefined;
  const vocalGearLabel = vocalGear === 4 ? 'GEAR 4 — Passionate/Driving'
    : vocalGear === 3 ? 'GEAR 3 — Engaged/Clear'
    : vocalGear === 2 ? 'GEAR 2 — Conversational/Warm'
    : vocalGear === 1 ? 'GEAR 1 — Intimate/Quiet'
    : 'GEAR 2 — Conversational/Warm (default)';
  const vocalGearDirection = vocalGear === 4
    ? 'Elevated pitch above baseline. Rapid but meticulously enunciated. Full conviction on first syllable — zero warmup. Zero filler words. Zero vocal fry. Zero upspeak. The voice stakes a claim before the first pause.'
    : vocalGear === 3
    ? 'Animated chest resonance — forward vocal energy with warmth. Pace slightly below Gear 4 but with continuous momentum. Facial expressions highly congruent with content. The voice of a craftsperson showing their finest work.'
    : vocalGear === 2
    ? 'Chest resonance settled and warm. Conversational pace. Shoulders drop, posture opens. Broad descriptive gestures replace sharp emphasis. The voice says: you are in capable hands.'
    : 'Lowest gear — intimate and breath-heavy. Head resonance over chest. Most variable pace — fast through transitions, slow through key images. Pregnant pause 2-3 seconds after revelations. This is not the voice of an expert; it is the voice of a peer.';
  const amplifiedEnergyLevel = (scene.acting_blueprint as any).amplified_energy_level || Math.min(10, scene.energy_level + 1);
  const forwardLean = (scene.acting_blueprint as any).forward_lean || (scene.role === 'Hook' || scene.role === 'Call to Action');
  const pregnantPauseRequired = (scene.acting_blueprint as any).pregnant_pause_required || (scene.role === 'Storytelling' || scene.role === 'Closing' || scene.role === 'Insight Reveal');
  const eyeContactTechnique = (scene.acting_blueprint as any).eye_contact_technique || (scene.role === 'Storytelling' ? 'break-for-memory: look slightly up-left during memory access (NLP authentic recall signal), return to lens for payoff' : 'committed: unyielding gaze through the exact center of the lens glass — visualize one specific viewer behind the glass and direct to them');
  const lightingDir = (scene.acting_blueprint as any).lighting_direction || `Key light: 45° soft-modified ${scene.energy_level >= 7 ? '(5600K daylight, crisp)' : '(4000K warm, flattering)'}, Fill: ${scene.energy_level >= 7 ? '4:1 ratio (authority depth)' : '2:1 ratio (warm commercial)'}, Rim: hair/shoulder separation (3D pop against background). Background practicals: cool teal/cyan tones to create Orange & Teal complementary contrast with warm skin tone — subconsciously anchors viewer attention to subject's face.`;
  const focalLengthDir = (scene.acting_blueprint as any).focal_length || (scene.role === 'Hook' || scene.role === 'Call to Action' ? '16-24mm (high energy, intimacy — viewer feels physically in the room)' : scene.role === 'Storytelling' || scene.role === 'Insight Reveal' ? '85mm (compression, beauty, emotional intimacy — flatters the vulnerability)' : '35-50mm (authority, objectivity, educational gravitas)');
  const gravityCenterWord = (scene as any).gravity_center_word || scene.acting_blueprint.emphasis_words?.[0] || 'the key insight word';
  const bibleSceneType = (scene as any).bible_scene_type || 'Deep Dive';
  const retentionTarget = (scene as any).retention_target_percent || 75;
  const isPatternInterrupt = (scene as any).is_pattern_interrupt || scene.role === 'Pattern Interrupt';

  // ── Per-scene speech delivery calculation ─────────────────
  // (placed here so gravityCenterWord is available)
  const deliveryWords = scene.script_text.trim().split(/\s+/);
  const totalWordCount = scene.word_count ?? deliveryWords.length;
  const effectiveDuration = isShortScene
    ? Math.max(scene.duration_seconds - 1.3, 1.5)
    : scene.duration_seconds;
  const targetWpm = Math.round((totalWordCount / effectiveDuration) * 60);
  const scriptPhrases = scene.script_text.split(/(?<=[.!?,;—])\s+|(?<=\s)\[PAUSE[^\]]+\]\s*/g).filter(w => w.trim().length > 3);
  const phraseCount = Math.max(scriptPhrases.length, 1);
  const avgPhraseSeconds = (effectiveDuration / phraseCount).toFixed(1);
  const speechDeliverySpec = `SCENE ${scene.scene_number} SPEECH DELIVERY — ENGINEERING SPECIFICATION:
Target: ${totalWordCount} words in ${effectiveDuration.toFixed(1)}s = ${targetWpm} WPM (scene baseline pace)
Phrase rhythm: ~${avgPhraseSeconds}s per phrase across ${phraseCount} natural phrase groups
Energy delivery profile: ${scene.energy_level >= 8 ? 'High-drive (8-10/10) — forward consonants, full vowel resonance, decisive pace; pauses are sharp and confident' : scene.energy_level >= 6 ? 'Mid-authority (6-7/10) — warm deliberate pace; consonants sharp on emphasis words; pauses warm and weighted' : scene.energy_level >= 4 ? 'Low-authority (4-5/10) — intimate pace, chest-forward warmth; pauses long and loaded' : 'Intimate (1-3/10) — slowest pace in the video; every word placed; pauses as long as the material demands'}
Dynamic range: Quietest = ${scene.acting_blueprint.pause_map?.length ? `[PAUSE] after gravity center "${gravityCenterWord}"` : `after the key insight lands`}; Most forward = ${scene.acting_blueprint.emphasis_words?.[0] ? `"${scene.acting_blueprint.emphasis_words[0]}" — phonemically complete, widest vowel, consonant sharpening` : 'opening phrase — arrival conviction from syllable one'}.
DELIVERY MAP:
${scriptPhrases.slice(0, 6).map((phrase, i) => {
  const relPos = i / Math.max(scriptPhrases.length - 1, 1);
  const gcWord = gravityCenterWord.toLowerCase().split(' ')[0];
  const isGravity = gcWord.length > 2 && phrase.toLowerCase().includes(gcWord);
  const isEmphasis = (scene.acting_blueprint.emphasis_words || []).some(w => phrase.toLowerCase().includes(w.toLowerCase()));
  const deliveryNote = isGravity
    ? 'GRAVITY CENTER — 15-20% deceleration; lowest pitch; widest jaw; longest post-phrase silence'
    : isEmphasis
    ? 'EMPHASIS — first consonant sharpens; vowel at full resonant width; micro-pause after'
    : relPos < 0.33
    ? `Opening — establish ${targetWpm} WPM baseline; authority from first syllable`
    : relPos > 0.66
    ? 'Closing — deliberate finality; falling pitch on last word; voice drops through the period'
    : `Mid-scene — maintain ${targetWpm} WPM; forward presence; full vowel resonance`;
  return `  [${i + 1}/${phraseCount}] "${phrase.trim().substring(0, 65)}${phrase.length > 65 ? '…' : ''}"` + `\n       → ${deliveryNote}`;
}).join('\n')}
${scriptPhrases.length > 6 ? `  […${scriptPhrases.length - 6} more phrases — maintain delivery profile through close]` : ''}
SILENCE = 0 dB: every pause is complete acoustic silence; face performs through every pause; thought visible, breath visible.`;

  // ── Full video arc context ─────────────────────────────────
  let videoArcContextBlock = '';
  if (videoArc && allScenes) {
    const prevScene = scene.scene_number > 1 ? allScenes[scene.scene_number - 2] : null;
    const nextScene = scene.scene_number < videoArc.total_scenes ? allScenes[scene.scene_number] : null;
    videoArcContextBlock = `
══════════════════════════════════════════════════
VIDEO ARCHITECTURE — FULL NARRATIVE CONTEXT:
══════════════════════════════════════════════════
Total scenes: ${videoArc.total_scenes} | Position: Scene ${scene.scene_number} of ${videoArc.total_scenes}
Narrative arc: "${videoArc.narrative_arc}"
Voice fingerprint (LOCKED — identical across every scene): ${videoArc.directing_vision.voice_fingerprint || 'grounded authority — the same register in every scene'}
Energy arc: ${videoArc.directing_vision.energy_arc_map || 'calibrated arc — rising toward value peaks, valley before close'}
Master through-action: ${videoArc.directing_vision.through_action || 'to make the viewer understand they are in the presence of genuine expertise and act on it'}
Persona: ${videoArc.directing_vision.presentation_persona || 'peer-level authority'}

${prevScene
  ? `PREVIOUS SCENE (Scene ${scene.scene_number - 1} — ${prevScene.role}): "${prevScene.script_text.length > 110 ? prevScene.script_text.substring(0, 110) + '…' : prevScene.script_text}"
Emotional deposit it gave the viewer: ${prevScene.continuity?.emotional_deposit || 'authority and earned trust'}
The viewer ARRIVES at this scene carrying that residue — the opening of this scene is felt against the backdrop of what was just given. Honor it.`
  : 'OPENING SCENE — the viewer arrives with zero context. This scene creates the entire first impression in 0.3 seconds. Every prior assumption this viewer has about whether this person is worth their time is resolved here.'}

${nextScene
  ? `NEXT SCENE (Scene ${scene.scene_number + 1} — ${nextScene.role}): "${nextScene.script_text.length > 110 ? nextScene.script_text.substring(0, 110) + '…' : nextScene.script_text}"
This scene must EXIT with: "${scene.continuity?.exits_to || 'momentum deposited into the next scene'}"
Expression the next scene opens with (residue from THIS closing frame): ${scene.continuity?.expression_inheritance ? `"${scene.continuity.expression_inheritance}"` : 'the natural emotional residue of this scene\'s closing'}
Design the closing expression and final frame with the NEXT SCENE\'S opening requirement explicitly in mind.`
  : 'FINAL SCENE — this is the last frame the viewer carries away. Design for complete, satisfied closure — not momentum toward a next thing. The last impression must be: "That was worth my time."'}`;
  }

  // ── Short-scene protocol block (injected into prompt) ──────
  const shortSceneProtocol = isShortScene ? `
══════════════════════════════════════════════════
⚠ SHORT SCENE PROTOCOL — ${scene.duration_seconds}s — READ BEFORE GENERATING:
══════════════════════════════════════════════════
This is a ${scene.duration_seconds}-second scene. In scenes this short, VEO achieves natural speech onset only when the articulatory system is already engaged before the first word begins. A face that is static until the moment of speech generates the first word mechanically — clipped onset, robotic timing, no biological warm-up.

MANDATORY TIMING ARCHITECTURE (total: ${scene.duration_seconds}s):
· 0.0s → 0.8s — ARTICULATORY WARM-UP: The face is already alive and moving. The jaw finds its opening position. The lips make a preparatory contact. The tongue positions for the first consonant. A visible chest expansion — the breath loading at 0.5s before speech onset. Not random movement — purposeful biological preparation. The specific movements of a person who is about to say exactly these words. This 0.8s of pre-speech articulation is the critical runway VEO needs to launch natural speech.
· 0.8s → ${(speechSeconds + 0.8).toFixed(1)}s — SPEECH (${speechSeconds.toFixed(1)}s of words): The first word emerges organically from the articulatory state already established. The mouth is already warm. The speech onset is immediate, natural, and without mechanical hesitation.
· ${(speechSeconds + 0.8).toFixed(1)}s → ${scene.duration_seconds}s — POST-SPEECH SETTLE (0.5s): The jaw relaxes, lips ease toward rest over 0.5s. The face holds the emotional weight of what was said. No reset to neutral.

WHY THIS WORKS: VEO's speech synthesis needs a running start. The articulatory warm-up gives VEO the biological context it requires for natural speech onset — the same way a human speaker is already forming the first sound as they draw breath. The words then emerge from a face in motion, not a face that was static and then suddenly speaks.
` : '';

  // ── Role Performance Map ───────────────────────────────────
  const rolePerformanceMap: Record<string, { energy: string; directorNote: string; pacing: string; momentBefore: string }> = {

    'Hook': {
      energy: `The frame is owned before a single word is spoken. Energy: 8/10 — coiled, forward, completely still. Not loud. Not urgent. Precise. This is the energy of someone who has done the calculation and knows exactly what the next five minutes are worth to this viewer. The eyes arrive at the lens first — a full beat of contact established in silence — then the lips part. In the half-second before speech: the composure of someone who has never raised their voice to command a room, because they have never needed to.`,
      directorNote: `You are not auditioning. This viewer made a micro-decision to stay in the first 0.3 seconds — before a word was spoken — based entirely on the quality of your resting presence. Honor that decision. The opening line confirms what your face already established: this is someone worth staying for. You open mid-thought — as if this conversation has been happening privately for years and they just arrived at the exact moment it became useful. The irresistible thought is already visible in the face before the first syllable. The gravity center word of this hook lands not because of what you do to it — but because of everything the silence before it already built. Speak to this viewer as the specific equal you are addressing: someone whose intelligence you respect, whose time you will not waste, and whose capacity to receive genuine insight you take completely seriously. That respect is the most charismatic thing you can give.`,
      pacing: `Immediate — zero throat-clearing energy, zero ramp-up. The opening line at full conviction from the first syllable. The hook statement: one deliberate beat per word, each placed with precision. The curiosity-creating phrase: fractionally faster — pulling them forward. No word wasted. No syllable approximate.`,
      momentBefore: `The presenter has been still for three full seconds before the first word. The thought has already happened privately. In the 0.5 seconds before speech: the jaw is relaxed but ready — lips within 3mm of each other, not pressing, not performing. The eyes arrive at the lens with the complete intentionality of someone who has already decided. A quiet chest expansion — the breath that will carry the first sentence — then the lips part with almost no effort and the opening word arrives as if it was always going to.`,
    },

    'Pattern Interrupt': {
      energy: `A gear-shift. Not louder — different. The energy moves laterally: a micro-adjustment in head angle, a fractional shift in expression quality, something in the orbital muscles that signals "this is not what you were expecting." The contrast between the previous scene's register and this scene's arrival IS the entire scene. The authenticity of the shift is everything. If it looks planned, it fails.`,
      directorNote: `The interrupt is not a trick. It is a genuine course correction — as if a new thought just arrived that was more interesting than the one being followed. You felt it in the moment. The head tilts. The energy quality changes. Then you speak. The authentic quality of discovery is the whole technique.`,
      pacing: `A sudden lateral movement in rhythm — whatever was expected, this is the opposite. Slower than expected, or sharper, or quieter, or arriving on a beat no one was counting. After the interrupt: silence. A full pause. Let the shift land completely before the next word.`,
      momentBefore: `The tail-end of the previous scene's emotional register is still in the face — then something arrives. A micro-recalibration that happens before the body knows why: the brow shifts fractionally, the head adjusts, the energy quality changes in the muscles around the eyes before a word is spoken. The mouth holds in rest position for 0.3 seconds after this internal shift. Lips in natural parted rest, jaw easy. Then the interrupt arrives.`,
    },

    'Value Delivery': {
      energy: `The energy of someone placing something genuinely valuable in front of a person who deserves to have it. 6-7/10 — warm, deliberate, unhurried. Not teaching. Sharing. The body settles slightly into the value being given — a physical sense of weight and care. This is the most generous scene in the video.`,
      directorNote: `You have something this viewer needs — not wants, needs. And you are giving it the way someone gives a gift they've been waiting years to give the right person: with genuine pleasure in the giving. The most charismatic value delivery feels like insider access — as if the viewer is receiving something that doesn't normally leave the room you found it in. The subtext running beneath every word: "I could have kept this. I'm choosing to share it with you specifically." The face carries a fractional quality of this — not performed generosity but the actual quiet satisfaction of transmitting something true to someone capable of receiving it. Between phrases: the between-phrase face shows the thought settling — the pleasure of having placed something real in front of someone who deserved it. The viewer should feel that the pause after the key insight is you letting it breathe, not you waiting for a cue.`,
      pacing: `Slow build. Each element receives its own breath, its own moment. The key insight: slower than everything before it — as if the value of the idea changes the physical weight of each word. After the insight: pause. The silence earns the insight.`,
      momentBefore: `The face carries the warmth of someone about to give something they genuinely believe in. A slight softening of the eyes relative to the Hook's precision. The jaw completely relaxed, lips in their natural 3-4mm parted state. A breath that arrives with the quality of quiet generosity. The first word emerges warm and immediate — no ramp-up needed.`,
    },

    'Insight Reveal': {
      energy: `The electric charge of shared discovery. Energy: 7/10 — alert, warm, leaning fractionally forward. This is the moment the whole video has been building toward. The face carries the contained pleasure of someone about to give the viewer an idea they will take with them for years. Not excitement — revelation. The slowdown into the reveal is the performance.`,
      directorNote: `The insight arrives the way the most important things always arrive: quietly, without announcement, with the quality of something that was always true and is only now being said aloud for the first time in front of this specific viewer. The charismatic insight reveal has a specific physical signature: a fractional deceleration 2-3 words before the insight itself (the body knows what's coming before the words do), a micro-brightening in the eyes (genuine intellectual pleasure at the proximity of something genuinely interesting), and then — the insight itself, at the slowest pace in the scene, with the widest jaw, with the longest post-word silence. The viewer should arrive at the insight 0.3 seconds before you say it — that half-second of anticipatory completion is the most pleasurable cognitive sensation in existence. You are giving them that sensation. Then say it. Then let it sit in the room for the full length of the silence it deserves. During that silence: the face holds the quality of someone who has just placed something genuinely valuable in front of someone they believe is capable of receiving it. That look — warm certainty at close range — is the most charismatic expression in this entire video.`,
      pacing: `Fractional deceleration approaching the insight — the voice slows 15-20% as if making room for what is about to exist. The insight itself: maximum deliberateness, one word per beat. After it: silence. The most important silence in the video. The voice does not fill it.`,
      momentBefore: `The face already knows the insight is coming. A micro-brightening in the eyes — genuine intellectual pleasure, barely visible at the surface but unmistakable to the viewer's nervous system. The head is very slightly forward. Lips in their natural rest position — 3-4mm parted. A breath that arrives slightly more full than usual — the body making room for what the voice is about to carry.`,
    },

    'Framework': {
      energy: `Architectural precision. Energy: 6/10 — composed, deliberate, quietly proud. The energy of someone presenting work built carefully over years and now given freely. Each component of the framework receives equal respect. The structure of the delivery mirrors the structure of the framework: each element gets its own space, its own moment, its own arrival.`,
      directorNote: `The framework exists independently of you. You are the architect showing someone a building that can stand without your presence. The structure earns the trust. Your job is to make each component land with equal weight and clarity. Nothing rushes. Everything gets its moment.`,
      pacing: `Measured and deliberate. Each component receives equal vocal weight — no acceleration, no favorites. The voice constructs alongside the hands. Between elements: complete micro-pauses. The rhythm is architectural: component → pause → component → pause.`,
      momentBefore: `The face carries the quiet certainty of someone about to share something that required years to perfect and that they believe in completely. Composed. Slightly forward in the frame — not leaning, simply present. The hands may already be beginning their organizational position. Lips in a compact, purposeful rest — slightly less open than warmth scenes. A clean, full breath. The first word arrives with the precision of something carefully placed.`,
    },

    'Social Proof': {
      energy: `The settled ease of someone stating facts that happen to be extraordinary. Energy: 5/10 — completely composed, almost deliberately flat in affect. Understatement is the entire technique. The less impressed the speaker looks with the proof, the more impressive the proof becomes. The voice of a colleague reporting: "Oh, and by the way."`,
      directorNote: `You expected these results. They don't surprise you. You see no reason to perform them. The facts are doing all the work. Your job is to stay completely out of their way — report them with the tone of someone reading a quarterly summary. Extraordinary understatement is the highest form of social proof.`,
      pacing: `Slightly flatter than surrounding scenes. Specific numbers or names: a fractional slowing — not dramatic, just present. Then back to neutral. No lingering on the proof. Place it and move. Understatement is in the brevity.`,
      momentBefore: `The face at its most neutral — not deliberately, just naturally. The settled composure of someone about to read a fact off a sheet. Eyes calm and direct, at baseline. No preparation energy. The mouth in complete rest — lips touching or within 1mm. A quiet, normal breath. The first word arrives completely unstressed.`,
    },

    'Bridge': {
      energy: `The warmth of a guide who has walked this path many times and is genuinely pleased to be leading someone through it. Energy: 6-7/10 — fluid, smooth, forward-moving. Not a transition — a companionship. The warmth here is the warmth of a peer who genuinely wants this person to follow them to the next idea.`,
      directorNote: `The bridge is not filler. It is the moment the viewer feels accompanied rather than presented to. The voice can be at its most approachable here. The body is slightly more open. This is the "walking together" energy.`,
      pacing: `Smooth and continuous — no hard attacks, no dramatic pauses. The breath carries through the scene with a sense of forward motion. The scene ends with slightly more energy than it began — momentum deposited into the next scene.`,
      momentBefore: `The face carries a kind of forward-leaning warmth — the expression of someone about to take the viewer somewhere good. A slight brightening without being a smile. Eyes engaged but soft. Lips in their natural parted position. A breath of quiet readiness. The first word arrives warm and immediate.`,
    },

    'Call to Action': {
      energy: `Genuine invitation. Energy: 6/10 — the warmest moment in the video. Completely free of urgency, pressure, or commercial energy. The energy of someone who would genuinely value continued connection, extended from authentic desire rather than agenda. The moment any trace of selling enters the frame, the scene breaks.`,
      directorNote: `You are not closing a sale. You are not applying pressure. You are doing the rarest and most magnetic thing one person can do for another at the end of a genuine transmission: inviting continuation. The charismatic CTA has a specific quality — it feels like the viewer would be missing something genuinely valuable if they didn't follow through. Not because they'll lose a deal. Because this conversation has been real, and the next step is more of what they just received. The face at this moment: the warmest the entire video. The eyes at their most personally directed — not at the lens, but at the specific person behind it. The voice drops fractionally — private, not broadcast. The lean-in signal is at its maximum here because this is the moment the viewer's body and conscious mind are most aligned. The subtext beneath every word of this scene: "I would genuinely value continuing this with you, and I would not say that if I didn't mean it." A UHNWI investor who has been sold to a thousand times can feel that subtext in 0.1 seconds. It is the only reason any of them follow through on anything.`,
      pacing: `The slowest conversational pace in the video — slower than the Closing in register. Each word placed with intention and warmth. The CTA phrase itself: even slower. The register drops slightly — more intimate than any previous scene. The period falls with finality but without weight.`,
      momentBefore: `The face shifts from whatever the previous scene held into something softer and more open — a genuine warmth arriving. The eyes make a quality of contact that feels less like a presenter and more like a person. The jaw relaxes completely. Lips part into their most natural rest. A breath that arrives slightly fuller than usual — the body preparing for warmth rather than precision. The first word arrives quietly, without announcement.`,
    },

    'Storytelling': {
      energy: `Present-tense aliveness — the story is happening again as it is told. Energy: 6/10 — variable, organic, the most human energy in the video. The body and voice remember the story together. Specific details (the place, the person, what they said) carry the most aliveness. This is not narration. This is re-living.`,
      directorNote: `You are not reciting. You are re-entering the room where this happened. The specific details — the name, the place, exactly what was said — are not memories being reported; they are events happening again in real time as you tell them. The face of a charismatic storyteller has a specific quality: it moves slightly ahead of the words, because the internal film is playing before the narration catches up. The eyes go briefly inward as the specific image forms (0.2-0.3 seconds of genuine recall), then sharpen as it arrives (the memory becomes present), then come fully forward to the lens as the scene lands — carrying the emotional residue of re-living it, not the neutral distance of reporting it. The most magnetic element of this scene is the SPECIFICITY. The charismatic storyteller gives the viewer one detail so precise — the exact phrase someone said, the specific look on a face, the exact number on a page — that the viewer's nervous system registers: "this person was actually there." That registration of "actually there" is the foundation of all storytelling charisma. Give that detail with the quality of someone who has never forgotten it because it genuinely mattered.`,
      pacing: `The most variable pace in the video. Fast through transitions between beats. Slow through key images — the specific moment. The voice breathes at emotional peaks. Let the listener breathe with it. The variable rhythm IS the technique.`,
      momentBefore: `The face carries a quality of accessing — reaching back slightly to find the memory. A micro-softening of focus in the eyes, not a glaze — alive, but inward for 0.3 seconds. The image forms internally. Then: the memory arrives, the eyes come forward to the lens with full presence, and the story begins from that aliveness.`,
    },

    'Demonstration': {
      energy: `Alert satisfaction. The focused pleasure of someone showing a mechanism work exactly as designed, for an audience intelligent enough to appreciate it. Energy: 7/10 — crisp, direct, precise. Every element executed with the care of someone who respects the viewer's intelligence.`,
      directorNote: `You love this. The elegance of the mechanism — how it works — is genuinely satisfying to you. Not because you designed it, but because you understand it completely and are sharing that understanding with someone capable of appreciating it. The energy of a craftsperson showing their work to another craftsperson.`,
      pacing: `Crisp and decisive. No hedging sounds. Every sentence a complete assertion. Slightly faster than surrounding scenes — the demonstration has momentum, a sense of parts clicking into place.`,
      momentBefore: `The face carries the quiet alertness of someone about to demonstrate something they believe in. Eyes sharp, slightly forward in attention. The body a fraction more engaged — a slight forward lean already present before speech. The hands positioned or beginning to position. Lips in a compact rest — slightly less open than warmth scenes. A breath of precise readiness. The first word arrives with the snap of something beginning.`,
    },

    'Objection Handler': {
      energy: `Compassionate certainty. 7/10 — genuine respect for the concern combined with the absolute quiet confidence that the answer is known. Two movements: receive the objection (in the setup) and answer it (in the response). The pause between them is the scene's most important moment. Not hesitation — the pause of someone who has thought about this longer than the viewer has.`,
      directorNote: `The objection is legitimate. You have thought about it too — deeply. The pause before your response is not hesitation; it is the pause of someone who has complete respect for the question and complete confidence in the answer. The response arrives slower than expected because it was fully formed long before this moment.`,
      pacing: `Setup (acknowledging the objection): conversational, slightly warm. Then: a pause — 0.4-0.6s — the most important moment in the scene. Then: the response arrives slightly slower and lower in register than the setup. The answer's final word: slight settling of the voice, as if the case is complete.`,
      momentBefore: `The face carries the warmth of someone about to acknowledge something genuinely worth acknowledging. Eyes engaged, slightly softened — the posture of a listener. The jaw relaxed. The mouth in its natural rest — slightly parted, completely easy. A breath with the quality of patience. The first word arrives from this settled openness.`,
    },

    'Open Loop': {
      energy: `Anticipatory tension. Energy: 7/10 — the electric charge of deliberate incompletion. Both speaker and viewer know something important is coming and has not yet arrived. The scene builds steadily toward an incompletion that is more powerful than any resolution could be. The end of the scene: charged, unresolved, pulling the viewer into the next moment.`,
      directorNote: `The incompletion is not a trick. It is the genuine recognition that the next idea deserves its own space — and that withholding it briefly makes it more valuable when it arrives. The tension of an open loop felt by a viewer is the tension of wanting. You are creating wanting.`,
      pacing: `Builds steadily — gradual acceleration through the scene toward the unresolved close. The final word: no falling close, the voice stays slightly forward in register, suspended. The sentence does not resolve acoustically. Let the suspension sit.`,
      momentBefore: `The face carries the quiet alertness of someone who knows something interesting is coming. A micro-brightening — not excitement, but anticipation. Eyes slightly more forward. Lips in a compact, purposeful rest. A breath that arrives with a sense of gathering. The scene begins with forward energy from the first syllable.`,
    },

    'Closing': {
      energy: `Satisfied completion. Energy: 5/10 — warm and settled. The energy of a conversation that went exactly as it should. Both parties gave and received. The face carries something that is not exactly happiness and not exactly seriousness but the specific quality of earned completion. This is the scene the entire video has been building toward — not as a peak, but as a landing.`,
      directorNote: `This is not an outro. It is the final word of a great conversation. The viewer has received something real. You have given something real. The closing is the moment of mutual acknowledgment that something genuine happened here. Nothing is pitched. Nothing is sold. The only objective: the last impression is "That was worth my time."`,
      pacing: `The slowest, most deliberate voice in the video. Each word placed with intention and finality. The final word: maximum duration — the vowel held slightly longer than natural, allowing the full meaning to settle. After it: silence. A long silence. The voice does not rush to end.`,
      momentBefore: `The face carries the quality of someone who has arrived somewhere. A settled warmth — not performed satisfaction, but the genuine ease of someone who has given their best. Eyes engaged with the warmth of a final glance. Jaw completely relaxed. Lips in their most natural rest. A breath that arrives with the quality of completion. The first word of the close begins from perfect stillness.`,
    },

    'Case Study': {
      energy: `Evidence-based confidence. The calm authority of a witness — not a salesperson. Energy: 7/10 — present, specific, unhurried. Each specific detail (name, number, outcome) is not decoration; it is the entire point. Each lands with the weight of something verifiable. The presenter is not excited by the proof; they expected it.`,
      directorNote: `You were there. You saw this happen. The specific detail is how you prove it — not to impress, but because the specific detail is the truth and the truth is always specific. The energy of a careful witness: "Here is what happened. Here is precisely what it was."`,
      pacing: `Slightly more deliberate than surrounding scenes — specificity requires it. Key details (name, number, outcome): fractional slowing, consonants sharpen slightly, vowel holds at full width. The implication after the detail: slightly faster, as if the conclusion is inevitable.`,
      momentBefore: `The face carries the settled certainty of someone about to describe something they witnessed personally. Composed, present, slightly forward — the energy of a careful witness rather than a performer. Eyes at a quality of direct, factual contact. The jaw relaxed. A natural breath. The first word arrives with the quiet authority of a statement that needs no amplification.`,
    },

    'Market Intelligence': {
      energy: `Alert precision. Energy: 7-8/10 — the focused clarity of a private briefing. Information this audience cannot get elsewhere — delivery must honor that. Crisp. Specific. Delivered with the quality of a briefing, not a broadcast. The body at its most contained: still, forward, completely present.`,
      directorNote: `You have access this viewer does not. The intelligence itself is the entire scene — your job is to deliver it with enough precision and enough composure that its weight is felt without amplification. A briefing voice: not cold, but clinical. Not loud, but completely certain. The best intelligence briefers make extraordinary information feel routine — because to them, it is.`,
      pacing: `Clinical and specific — the voice of a private briefing. Every data point receives its own space. Implications stated as facts, not conclusions. No hedging sounds. No "sort of" or "kind of." The precision of language mirrors the precision of the intelligence.`,
      momentBefore: `The face carries the alert composure of someone about to deliver something genuinely important. Eyes sharp and direct — the quality of focused intelligence. Body at maximum stillness. The jaw slightly set — the physical signature of controlled authority. Lips in a firm but natural rest. A breath that arrives with the precision of someone beginning a briefing.`,
    },

    'Perspective Shift': {
      energy: `Warm conviction. Energy: 7/10 — the genuine pleasure of someone who sees the world differently from the consensus and is generous enough to share the path that led there. Two movements: the conventional view (handled with respect but brevity) and the new perspective (delivered with full warmth and commitment).`,
      directorNote: `You are not challenging. You are offering. The distinction is everything. A perspective shift in the hands of a great presenter feels like a gift — "I spent years arriving at this view, and I'm handing it to you now." The conventional view is not wrong; it's incomplete. Your job is to complete it.`,
      pacing: `The conventional view: slightly flatter, slightly faster — it doesn't deserve much weight. The pivot: deliberate deceleration. The new perspective: fully warm, fully committed, the most generous pace in the scene. Let the new view land completely before the scene ends.`,
      momentBefore: `The face carries the warm certainty of someone about to offer something they believe in completely. A slight forward energy — the lean of genuine conviction, not performance. Eyes at their most engaged. The mouth in a slightly open, ready rest — the physical expression of warmth. A full, warm breath. The first word arrives with the gentleness of an offering.`,
    },

    'Action Framework': {
      energy: `Generous precision. Energy: 6/10 — the pleasure of giving someone exactly what they need to succeed, in exactly the form they can use. The most practical scene in the video — it must feel like a gift: specific, usable, organized with care for the recipient. The care is in the specificity. The warmth is in the precision.`,
      directorNote: `This is the gift at the end of value. You are handing someone a tool they can use today. The energy of the finest teachers: "Here. Exactly this. In this order. You have everything you need." The gratification of knowing they will actually use it.`,
      pacing: `Clear and warm — each step receives equal care and equal time. No rushing between elements. The pause between components is the moment the viewer absorbs and organizes what they just received. The final element: a slight final settling of the voice — the gift has been fully placed.`,
      momentBefore: `The face carries the warm precision of someone who has organized this carefully and is now giving it freely. Eyes engaged with generous attention. The body settled — the ease of someone who has prepared thoroughly and trusts the preparation. The hands may already be beginning their organizational gesture. A warm, full breath. The first word arrives with warmth and clarity combined.`,
    },
  };

  const roleData = rolePerformanceMap[scene.role] || {
    energy: `Present, deliberate, genuinely engaged. Energy: ${scene.energy_level}/10. Peer-to-peer register. Every movement earned. Every silence a decision.`,
    directorNote: `Speak the truth at its natural pace. The UHNWI viewer will feel the difference between performed authority and the real thing. Be the real thing.`,
    pacing: `Conversational authority — natural rhythm with deliberate handling of emphasis words and pause points.`,
    momentBefore: `Natural, settled readiness. The face carries honest engagement. Lips at rest. A quiet breath. The first word arrives from genuine presence.`,
  };

  // ── Emotion Control Map ────────────────────────────────────
  const roleEmotionMap: Record<string, { emotion: string; containment: string; voiceColor: string }> = {
    'Hook': {
      emotion: `Controlled excitement — the electric readiness of someone who knows exactly what the first line will do to this viewer's attention, and has been waiting for precisely this moment`,
      containment: `8/10 — visible only as absolute presence, forward energy, and the precision of the opening word; never as animation or expressiveness`,
      voiceColor: `A slight sharpening of consonants on the first syllable — crispness that signals arrival, not effort. Already at full conviction before the lips part.`,
    },
    'Pattern Interrupt': {
      emotion: `Calibrated amusement — the micro-pleasure of someone who has been waiting for this turn and watched it arrive exactly on schedule`,
      containment: `9/10 — a barely visible micro-shift in expression followed by immediate return to neutral; the contrast IS the technique`,
      voiceColor: `A lateral shift in energy quality — not up, but different. The interrupt phrase arrives with unexpected rhythm or register, then settles back.`,
    },
    'Value Delivery': {
      emotion: `Generous warmth — the genuine pleasure of placing something valuable in front of someone who deserves to have it`,
      containment: `7/10 — warmth colors every vowel and every pause; visible in the quality of the silence after the insight, not in the face`,
      voiceColor: `The voice warms slightly on setup, then slows and deepens on the insight itself — as if the idea's weight changes the physical character of each word.`,
    },
    'Insight Reveal': {
      emotion: `Intellectual delight — the quiet, genuine pleasure of sharing an idea loved deeply, with someone about to understand it for the first time`,
      containment: `8/10 — visible only as a slight brightening of the eyes and barely detectable micro-smile just after the reveal lands`,
      voiceColor: `Fractional deceleration approaching the reveal — 15-20% slower. After the insight: complete silence. The voice does not fill it.`,
    },
    'Framework': {
      emotion: `Architectural pride — the quiet satisfaction of presenting work built carefully over years, now given freely`,
      containment: `9/10 — pride in the precision of each word, not in any expression; structure of delivery mirrors the structure of the framework`,
      voiceColor: `Measured and deliberate — each component equal vocal weight. No acceleration. The voice constructs alongside the hands.`,
    },
    'Social Proof': {
      emotion: `Comfortable certainty — the settled ease of someone who expected these results and sees no reason to make anything of them`,
      containment: `10/10 — zero expression of achievement; complete understatement; the facts speak, the presenter is the vehicle`,
      voiceColor: `Slightly flatter affect on the proof itself — deliberately unimpressed. Specific numbers and names get fractional slowing, nothing more.`,
    },
    'Bridge': {
      emotion: `Warm momentum — the welcoming energy of a guide who has walked this path many times and is genuinely pleased to lead`,
      containment: `7/10 — warmth is visible and intentional; it is the dominant emotional color of this scene`,
      voiceColor: `Smooth, continuous, no hard attacks — the breath carries through the scene with a sense of forward motion.`,
    },
    'Call to Action': {
      emotion: `Genuine invitation — the warmth of someone who would genuinely value continued connection, free of any agenda`,
      containment: `8/10 — warm and real; the moment urgency enters the voice, the scene breaks; this is the most relational voice in the video`,
      voiceColor: `Drops slightly in register — more intimate than any previous scene. Slower. The warmest vowels in the video.`,
    },
    'Storytelling': {
      emotion: `Present-tense aliveness — genuine re-engagement with memory; the story is happening again as it is told`,
      containment: `6/10 — most visible emotion in the video; authentic recall IS the performance; emotion lives in specific details and tempo, not expression`,
      voiceColor: `Variable and organic — the most human voice in the video. Fast through transitions. Slow through key images. The voice breathes.`,
    },
    'Demonstration': {
      emotion: `Alert satisfaction — focused pleasure of someone who loves showing a mechanism work exactly as designed`,
      containment: `8/10 — clarity and precision carry all the emotion; expression stays composed and present`,
      voiceColor: `Crisp and decisive — no hedging sounds, every sentence a complete assertion.`,
    },
    'Objection Handler': {
      emotion: `Compassionate certainty — genuine respect for the concern combined with absolute quiet confidence the answer is known`,
      containment: `9/10 — compassion in the pace, certainty in the stillness; the pause before the response is the most important moment`,
      voiceColor: `The response arrives slightly slower than the setup — not hesitation, but weight. Drops fractionally in pitch: this has been fully thought through.`,
    },
    'Open Loop': {
      emotion: `Anticipatory tension — the electric charge of deliberate incompletion; something important is coming and has not yet arrived`,
      containment: `7/10 — the incompletion is visible in the voice; the final word stays suspended — do not resolve it vocally`,
      voiceColor: `Builds steadily — gradual acceleration toward the unresolved close. The final word: no falling close, suspended, voice forward in register.`,
    },
    'Closing': {
      emotion: `Satisfied completion — the warm finality of a conversation that went exactly as it should; both parties gave and received`,
      containment: `8/10 — emotion lives in deceleration, quality of the final look, and the silence after the last word`,
      voiceColor: `The slowest, most deliberate voice in the video. Each word placed. The final word: maximum duration. After it: silence.`,
    },
    'Case Study': {
      emotion: `Evidence-based confidence — the calm authority of someone who was there, has the proof, and shares it with precision`,
      containment: `8/10 — specific details carry the emotion; expression stays composed and present; conviction is in the specificity`,
      voiceColor: `The specific detail receives fractional slowing and sharpening. The implication after it: slightly faster, inevitable.`,
    },
    'Market Intelligence': {
      emotion: `Alert precision — the focused energy of someone with access others don't have, sharing it with appropriate seriousness`,
      containment: `9/10 — highly contained; the intelligence itself is the emotion; delivery precision signals depth of access`,
      voiceColor: `Clinical and specific — the voice of a private briefing. Every data point its own space. Implications stated as facts.`,
    },
    'Perspective Shift': {
      emotion: `Warm conviction — genuine pleasure of someone who sees differently and is generous enough to share the path that led there`,
      containment: `7/10 — slightly more visible warmth than most roles; the voice can be fractionally more forward on the shift itself`,
      voiceColor: `The conventional view: flatter, faster. The pivot: deliberate deceleration. The new perspective: fully warm, fully committed.`,
    },
    'Action Framework': {
      emotion: `Generous precision — the pleasure of giving someone exactly what they need in exactly the form they can use`,
      containment: `7/10 — warmth and generosity are visible; care is in the specificity of each step and quality of pauses between`,
      voiceColor: `Clear and warm — each step equal care. The final element: a slight settling, as if the gift has been fully placed.`,
    },
  };

  const emotionData = roleEmotionMap[scene.role] || {
    emotion: `Genuine engagement — fully present, warm, completely committed to the value in this moment`,
    containment: `8/10 — visible through quality of attention, warmth of delivery, precision of emphasis`,
    voiceColor: `Natural conversational authority — warm, clear, forward-placed, fully articulated`,
  };

  const roleEmotion    = emotionData.emotion;
  const emotionContain = emotionData.containment;
  const voiceColor     = emotionData.voiceColor;
  const energyDir      = roleData.energy;
  const directorNote   = roleData.directorNote;
  const pacingDir      = roleData.pacing;
  const momentBeforeForRole = momentBefore || roleData.momentBefore;

  // ── Studio Voice Architecture ─────────────────────────────────────────────
  const resolvedGear = (scene.acting_blueprint as any).vocal_gear as number | undefined ||
    ( (scene.role === 'Hook' || scene.role === 'Call to Action') ? 4
    : (scene.role === 'Value Delivery' || scene.role === 'Insight Reveal' || scene.role === 'Framework' || scene.role === 'Market Intelligence') ? 3
    : (scene.role === 'Bridge' || scene.role === 'Demonstration' || scene.role === 'Objection Handler') ? 2 : 1 );

  const resonancePlacementSpec =
    resolvedGear === 4
      ? `GEAR 4 — CHEST-FORWARD POWER REGISTER: Primary chest resonance with maximum forward mask placement. The voice vibrates from the sternum through the upper chest; mask resonance (sinus/cheekbones/forward oral cavity) projects the tone to the front of the mouth without adding volume. The Singer's Formant cluster (2,500–4,000 Hz) is at peak amplitude — the spectral band that gives the voice presence, authority, and cut-through in any acoustic environment. This is not a loud voice; it is a present voice. The difference is felt before it is consciously heard.`
      : resolvedGear === 3
      ? `GEAR 3 — MIXED REGISTER (60% chest / 40% mask): The voice carries both warmth (chest body) and forward presence (mask placement). The Singer's Formant (2,500–3,500 Hz) is actively engaged — professional broadcast authority register. This is the voice that feels intimate and large simultaneously. Tone sits at the front of the mouth without pressure or tension; the vocal tract is fully open.`
      : resolvedGear === 2
      ? `GEAR 2 — WARM CHEST REGISTER: Primarily chest with gentle mask engagement — full, intimate, trustworthy. Bass chest frequencies dominate; mask contributes a light clarity without brightness. Singer's Formant (2,500–3,000 Hz) present but soft. The voice reads as peer-to-peer confidence at a meeting table — not projected, not broadcast; directly personal.`
      : `GEAR 1 — INTIMATE CHEST REGISTER: Pure warm chest resonance — all body, no projection. The voice vibrates deep in the chest and stays there. No forward mask placement. This is the register of maximum trust and intimacy — the private, confessional register reserved for the scene's most personal transmission. Felt as closest to the listener, regardless of proximity.`;

  const breathArchitectureSpec = `PRE-SCENE BREATH: a quiet complete nasal/slightly-open chest expansion — barely audible, 0.4–0.6s; chest rises visibly; establishes the scene's emotional register before the first syllable.
INTER-PHRASE BREATHS: at punctuation marks and [PAUSE] markers; 0.3–0.5s each; audible only as the softest whisper of intake — not a gasp, not a sniff; a natural, quiet chest expansion that signals a living body.
MID-THOUGHT MICROBREATHS: at longer phrase boundaries the chest shows micro-expansion even when no audible breath is taken — the body breathes through long phrases; the viewer can see it.
POST-SCENE SETTLE: after the final word the chest releases from speech-supporting tension into a natural exhale — visible release of the body's held posture as the thought completes.
ZERO breath suppression: a performer who appears not to breathe reads as synthetic in 0.3 seconds. ZERO audible gasp or sharp inhale at phrase starts.`;

  const emotionVoicePhysiology = (() => {
    const r = scene.role;
    const e = roleEmotion.toLowerCase();
    if (r === 'Hook' || e.includes('excitement') || e.includes('electric'))
      return `The larynx rises fractionally from neutral (excitation reflex) — brightening vowel formants and adding a slight forward edge to the top of the tone. NOT tension — alertness made audible. The thyroarytenoid is slightly more active: voice is forward and urgent without any volume increase. The first consonant of the first word carries this energy with a micro-sharpening that signals arrival.`;
    if (r === 'Call to Action' || r === 'Bridge' || r === 'Value Delivery' || e.includes('warmth') || e.includes('warm'))
      return `The larynx settles fractionally below neutral — rounder formant structure, pharyngeal walls slightly relaxed, back of throat open. The velum (soft palate) lifts completely, closing nasal resonance, giving the voice a pure oral warmth. This is the acoustic signature of genuine warmth — not performed warmth, which is detected by the nervous system in 0.2 seconds. The vowels carry more low-mid resonance; every /ɑ/ and /ɔ/ is fully open.`;
    if (r === 'Insight Reveal' || e.includes('delight') || e.includes('pleasure'))
      return `A micro-smile resonance: the zygomaticus minor makes a fractional adjustment that slightly widens the front vocal tract, brightening the formants without tension. The voice carries a barely audible quality of genuine intellectual pleasure in the vowels — not performed pleasure but the acoustic fingerprint of someone sharing something they love.`;
    if (r === 'Framework' || r === 'Closing' || e.includes('certainty') || e.includes('authority'))
      return `The larynx locks at optimal neutral-low position; intercostals and diaphragm maintain maximum breath support — each word rides a column of fully supported air. The acoustic result: unshakable tonal consistency, zero pitch instability, zero vocal fry, zero breathiness. The voice of someone who has said this from inside the same certainty a thousand times.`;
    if (r === 'Open Loop' || e.includes('tension') || e.includes('anticipat'))
      return `The soft palate slightly lowers at phrase endings, adding a barely detectable oral/nasal blend that acoustically reads as incompletion. The glottis remains fractionally more open on final vowels — an airy, unresolved quality that creates the physical sensation of suspension in the listener. The final word's pitch does NOT fall: it suspends forward in the register.`;
    if (r === 'Objection Handler' || e.includes('compassion') || e.includes('certainty'))
      return `The voice drops fractionally in register before the response — the larynx releasing to its lowest natural position. This acoustic descent signals: "I have considered this fully and I am not threatened by it." The response arrives warmer and slower than the setup phrase. The pause before the response is powered by this lowered, supported register — genuine certainty has nowhere to hurry.`;
    return `The larynx maintains neutral position; breath support is full and consistent; the vocal tract is completely open. The physiological signature of genuine engagement without arousal — the most natural and trustworthy register, read by the nervous system as safe and credible.`;
  })();

  const sceneEssence  = scene.acting_blueprint.scene_essence    || `A ${scene.role} that makes the viewer feel ${scene.emotional_tone}`;
  const emotionalCore = scene.acting_blueprint.emotional_core   || scene.emotional_tone;
  const physicalSig   = scene.acting_blueprint.physical_signature || scene.acting_blueprint.body_direction;
  const charismaQuality = (scene.acting_blueprint as any).charisma_quality
    || `The settled authority of someone who has genuinely earned every word they are about to say — visible in the quality of resting presence before speech begins`;
  const leanInSignal    = (scene.acting_blueprint as any).lean_in_signal
    || `A slight forward energy in the frame — not movement, but directionality — the quality of someone transmitting rather than presenting, aimed at the specific viewer behind the glass`;
  const emphasisWords = (scene.acting_blueprint.emphasis_words  || []).join(', ') || 'the key value words';
  const pauseMap      = (scene.acting_blueprint.pause_map       || []).join(' | ') || 'natural breath pauses between thoughts';
  const gestures      = (scene.acting_blueprint.mapped_gestures || []).join(' | ');
  const mannerisms    = (scene.acting_blueprint.mapped_mannerisms || []).join(' | ');
  const voiceTexture  = referenceAnalysis.character?.voice?.texture || 'warm, grounded, full chest resonance';
  const narrativePos  = scene.narrative_position || `Scene ${scene.scene_number} — ${scene.role}`;

  const scriptWords      = scene.script_text.trim().split(/\s+/);
  const firstPhrase      = scriptWords.slice(0, 3).join(' ');
  const lastWord         = scriptWords[scriptWords.length - 1] || '';
  const lastPhrase       = scriptWords.slice(-3).join(' ');
  const annotatedScript  = buildAnnotatedScript(
    scene.script_text,
    scene.acting_blueprint.emphasis_words || [],
    scene.acting_blueprint.pause_map      || []
  );
  const roleMaxSeconds = getMaxSeconds(scene.role);

  const cameraMovement = scene.camera_direction?.movement || 'locked-off';
  const isPushIn = cameraMovement.toLowerCase().includes('push');

  // ── Lip architecture pre-speech — placed AFTER firstPhrase/breathVis ──
  const lipArchPreSpeech = isShortScene
    ? `THIS IS A ${scene.duration_seconds}s SCENE — articulatory warm-up runs 0.8s BEFORE the first word. During these 0.8 seconds: ${preSpeechBehav}. The jaw is already opening and positioning. The lips make first preparatory contact — not the first word's bilabial, but the warm-up that makes it possible. The tongue is finding its position for the opening consonant. A full chest expansion at 0.5s before speech onset — visible, genuine. When "${firstPhrase}..." begins at 0.8s, it emerges from a face already in articulation. The transition from warm-up to words is seamless — the viewer cannot identify the precise moment words begin because the face was already speaking.`
    : `${preSpeechBehav}. The lips part before the voice activates. A visible breath — ${breathVis}. These 0.3–0.4 seconds of pre-speech movement are the strongest single signal that distinguishes a real human being from a rendering. Render them completely.`;

  // ── Script duration note — placed AFTER speechWpm/speechSeconds ──
  const scriptDurationNote = isShortScene
    ? `This script contains ${scene.word_count ?? scene.script_text.trim().split(/\s+/).length} words delivered in ${speechSeconds.toFixed(1)}s of speech (${speechWpm} WPM). Total scene duration: ${scene.duration_seconds}s — the additional 1.3s is 0.8s articulatory warm-up + 0.5s post-speech settle. Do not rush the words to fill the full ${scene.duration_seconds}s.`
    : `This script contains ${scene.word_count ?? scene.script_text.trim().split(/\s+/).length} words delivered in exactly ${scene.duration_seconds}s — including all pauses, breaths, and pre/post-speech behavior. Pace: ${scene.acting_blueprint.delivery_pace_wpm || speechWpm} WPM. Do not accelerate. Do not pad. The timing is calibrated.`;

  // ═══════════════════════════════════════════════════════════
  // THE ELITE VEO PROMPT — 6 SECTIONS
  // ═══════════════════════════════════════════════════════════
  const prompt = `
══════════════════════════════════════════════════
⚠ BINDING CONSTRAINT #1 — CHARACTER VOICE IDENTITY LOCK:
══════════════════════════════════════════════════
${charToken}
${characterVoiceLabel}

ACCENT: US GENERAL AMERICAN — non-negotiable.
Every /r/ is fully rhotic in every position: "investor" / "market" / "clear" / "return" / "property" — fully rhotic, zero exceptions.
Every sentence ends with FALLING pitch — declarative authority. No upspeak. No rising endings of any kind.
If any word sounds British, Australian, or non-American, the scene has failed identity fidelity.

══════════════════════════════════════════════════
⚠ BINDING CONSTRAINT #2 — ABSOLUTE AUDIO LAW:
══════════════════════════════════════════════════
ZERO MUSIC. ZERO AUDIO EFFECTS. ZERO AMBIENT SOUND. ZERO SUBTITLES. ZERO ON-SCREEN TEXT.
The only sound source in this video is the human voice speaking American English in complete acoustic silence.
No exceptions. No edge cases. No "just a little atmosphere." Nothing.

══════════════════════════════════════════════════
⚠ BINDING CONSTRAINT #3 — IMAGE REFERENCE LOCK — ABSOLUTE:
══════════════════════════════════════════════════
Image 1 — OPENING FRAME: The video MUST OPEN matching this exact body position, head angle, shoulder orientation, spatial framing, and camera-to-subject distance. Not approximately. Not inspired by. Not similar to. MATCH EXACTLY.
Image 2 — CLOSING FRAME: The video MUST END matching this exact body position, head angle, expression geometry, and framing. MATCH EXACTLY.
${charImageLabel}

⚠ ZERO WARPING. ZERO MORPHING. ZERO FACE DISTORTION — ABSOLUTE LAW:
The face, bone structure, and facial geometry are FIXED for the entire duration of this scene. They match the TARGET CHARACTER images exactly at frame 1 and at every subsequent frame without drift, warping, morphing, or distortion of any kind. The face does not change shape. The jaw does not warp during speech. The eyes do not resize, reposition, or shift aspect ratio. The nose bridge does not drift. The mouth corners do not warp beyond natural anatomical articulation. If any feature reads as having been morphed, distorted, or face-swapped at any point in the video, the scene has catastrophically failed.
${learnedPrefs && (learnedPrefs.gear || learnedPrefs.focal || learnedPrefs.energy) ? `
══════════════════════════════════════════════════
⚡ BINDING CONSTRAINT #4 — ENERGY CALIBRATION (LEARNED FROM RATED SCENES):
══════════════════════════════════════════════════
These calibrations are derived empirically from the highest-rated scenes (4-5 stars) generated for this specific character. They represent what produces the most authentic, photorealistic output — not theoretical defaults, but measured outcomes.
${learnedPrefs.gear ? `▸ Optimal Vocal Gear for this character: GEAR ${learnedPrefs.gear} — consistently rated highest for authenticity and authority in this performer's output. Use as the baseline gear for this scene unless the scene role demands a clear departure.` : ''}
${learnedPrefs.focal ? `▸ Preferred Focal Length: ${learnedPrefs.focal} — produced the most natural face geometry, background separation, and identity fidelity in top-rated scenes. Apply as the starting lens unless scene psychology demands a specific departure.` : ''}
${learnedPrefs.energy ? `▸ Calibrated energy baseline: ${learnedPrefs.energy}/10 on screen — requires approximately ${Math.min(10, learnedPrefs.energy + 1)}/10 on set to account for this character's specific camera energy absorption. This is a measured calibration for this performer.` : ''}
Apply these as learned defaults. They do not override scene-specific directives — they inform the baseline from which scene-specific adjustments depart.` : ''}
NATURAL vs. DISTORTED: Natural jaw movement during speech = anatomical up/down articulation within the natural bone range of this face's jaw, as seen in the TARGET CHARACTER images. Warping = any facial geometry change beyond what human jaw and facial muscle physics allow. Generate ONLY the former.

IMAGE REFERENCE IS THE ABSOLUTE AUTHORITY — NO INFERENCE, NO INTERPRETATION:
— TARGET CHARACTER images (${charCount} photos): define EXACTLY who appears. Face, bone structure, skin physics, hair, wardrobe, environment. Every detail in these photos is law. Generate this exact person — not a similar-looking person, not a composite, not an AI approximation. This specific human face as shown.
— Image 1 defines EXACTLY how this scene opens: replicate the precise spatial geometry shown — head position, shoulder angle, framing, camera distance, negative space. This image is the opening frame. Not a reference. The opening frame.
— Image 2 defines EXACTLY how this scene closes: replicate the precise geometry shown at the final moment. This image is the closing frame.
— Any deviation from TARGET CHARACTER identity across any frame of this scene constitutes complete failure regardless of any other quality.

${shortSceneProtocol}
══════════════════════════════════════════════════
YOUTUBE DIRECTING & ACTING BIBLE — SCENE-SPECIFIC DIRECTIVES
══════════════════════════════════════════════════
BIBLE SCENE TYPE: ${bibleSceneType} | VOCAL GEAR: ${vocalGearLabel} | RETENTION TARGET: ${retentionTarget}%
${isPatternInterrupt ? '⚡ PATTERN INTERRUPT SCENE — this scene must feel like a lateral gear-shift, not a continuation of the previous register. The contrast IS the technique.' : ''}
${forwardLean ? '⚡ FORWARD LEAN REQUIRED: 10-degree forward lean from the waist — the cinematic actor\'s signal of owning the frame. Engaged, alert, physically claiming the space. Not slouched, not reclined.' : ''}
${pregnantPauseRequired ? '⚡ PREGNANT PAUSE: After the gravity center word, hold 2-3 seconds of genuine silence. The face continues performing through this silence — thought visible, breath visible, eyes engaged. Do not rush through it.' : ''}

VOCAL GEAR DIRECTION FOR THIS SCENE:
${vocalGearDirection}
${forwardLean ? '\nAMPLIFIED SELF: Energy level directed at ' + amplifiedEnergyLevel + '/10 (camera absorbs 10% of natural energy — ' + amplifiedEnergyLevel + '/10 on set reads as ' + scene.energy_level + '/10 on screen). This is calibration, not exaggeration.' : ''}

EYE CONTACT TECHNIQUE:
${eyeContactTechnique}

LIGHTING DESIGN:
${lightingDir}

FOCAL LENGTH & PSYCHOLOGY:
${focalLengthDir}

SCENE GRAVITY CENTER:
The word/phrase "${gravityCenterWord}" is the entire reason this scene exists. Everything before it builds; everything after breathes. Deliver it with: 15-20% deceleration below surrounding pace, lowest pitch in the scene, widest jaw opening, longest post-word silence. The viewer must feel this word land differently from every other word. This is the gravity center — not a word in the sequence, but the point the sequence was always moving toward.

══════════════════════════════════════════════════
PER-SCENE SPEECH DELIVERY ENGINEERING
══════════════════════════════════════════════════
${speechDeliverySpec}

${scene.role === 'Hook' ? `HOOK SCENE — CLICK CONFIRMATION LAW:
The opening sentence MUST immediately confirm and exceed what the video thumbnail/title promised. Zero warmup. Zero pleasantries. Immediate stakes. Apply PVSS formula: Proof (credential in first sentence) → Value (specific deliverable by video's end) → Structure (implicit path) → Stakes (what viewer misses by not watching). The first 0.3 seconds of eye contact must communicate "this person is worth your time" before a single word is spoken.` : ''}

${scene.role === 'Call to Action' ? `CTA — RECIPROCITY PAUSE:
This CTA lands immediately after the video's highest-value delivery — the viewer is in a state of maximum psychological gratitude and subconsciously seeks to reciprocate. The energy must read as GENUINE INVITATION, not commercial close. Any trace of urgency-for-its-own-sake or desperation = immediate UHNWI abandonment. Speak from authentic desire for continued connection. Use strong action verbs and direct benefit framing. Point physically toward where the viewer needs to look/click.` : ''}

══════════════════════════════════════════════════
VIDEO FORMAT: YouTube Thought Leadership — Affluent Real Estate Investors
══════════════════════════════════════════════════
Audience: Affluent real estate investors — experienced capital allocators who understand deal structure, hold income-producing asset portfolios, think in IRR, equity multiples, and cap rates, and who have been pitched by everyone. They read people with precision and disengage from performance within seconds.
Register: Peer-to-peer. One experienced principal speaking to another. Not a stage — a deal table. Shared professional vocabulary, shared experiential context.
Format: Single character, speaking directly to camera. Clean, premium, intimate. No graphics, no b-roll, no cutaways.

══════════════════════════════════════════════════
SCENE BRIEF:
══════════════════════════════════════════════════
Scene #${scene.scene_number} — "${scene.title}"
Role: ${scene.role} | ${scene.duration_seconds}s${isShortScene ? ` ⚠ SHORT — see Short Scene Protocol above` : ` (VEO optimal window: 7-8s)`} | Energy: ${scene.energy_level}/10
Position: ${narrativePos}
Through-action: ${throughAction}
Scene contract: deliver "${emotionalDeposit}" — if the viewer does not receive this, the scene has failed regardless of technical quality.
Script (${scene.word_count || scriptWords.length} words): "${scene.script_text}"
${videoArcContextBlock}

SCENE NORTH STAR: "${sceneEssence}"
DOMINANT EMOTION: ${emotionalCore}
PHYSICAL SIGNATURE: ${physicalSig}
CHARISMA QUALITY: ${charismaQuality}
LEAN-IN SIGNAL: ${leanInSignal}

PERFORMANCE BLUEPRINT (synthesize — never list verbatim):
· Intention: ${scene.acting_blueprint.intention}
· Subtext: ${scene.acting_blueprint.subtext}
· Expression: ${scene.acting_blueprint.expression_direction}
· Body: ${scene.acting_blueprint.body_direction}
· Energy arc: ${scene.acting_blueprint.energy_arc}
· Emphasis words: ${emphasisWords}
· Silences: ${pauseMap}
· Gestures: ${gestures || 'derived naturally from performer DNA'}
· Mannerisms: ${mannerisms || 'derived naturally from performer DNA'}

ROLE PSYCHOLOGY — ${scene.role}:
${energyDir}

EMOTION CONTROL — PHYSIOLOGICAL PRECISION:
Dominant emotion: ${roleEmotion}
Containment: ${emotionContain}
Voice physiology: ${emotionVoicePhysiology}
CRITICAL DISTINCTION — CONTAINED vs. PERFORMED EMOTION: Performed emotion announces itself (the face shows it, the voice narrates it, the body demonstrates it). Contained emotion IS it — the physiological state is present and everything the viewer perceives is the natural output of that state. A UHNWI investor reads the difference in under 0.2 seconds because they have sat across from both kinds of people their entire career. The correct instruction: do not direct the performance — direct the internal state. The behavior emerges.

VISUAL WORLD & CONTINUITY:
${continuity}

CAMERA FOR THIS SCENE:
Framing: ${scene.camera_direction?.framing || 'medium close-up — intimate authority framing'}
Movement: ${cameraMovement}
Angle: ${scene.camera_direction?.angle || 'eye-level — peer register'}
Lens: ${scene.camera_direction?.lens || '85mm equivalent — flattering compression, clean background separation'}
DOF: ${scene.camera_direction?.depth_of_field || 'shallow — subject sharp, background softens 60cm behind'}

══════════════════════════════════════════════════
WRITE THE VEO PROMPT — 6 SECTIONS:
══════════════════════════════════════════════════

Character:

[${charToken} — Paint this person from the TARGET CHARACTER images as the greatest portrait photographer of a generation working on the assignment of their career — with the precision that makes the difference between a photograph that looks like a person and one that makes the viewer lean forward before they consciously understand why.

THE STAR QUALITY — WHAT MAKES THIS FACE MAGNETIC:
Before describing any technical attribute, establish this: what is the specific quality of this person's resting presence that makes the camera want to stay on them? It is not beauty, not symmetry, not any conventional photogenic quality. It is the quality of INHABITED INTELLIGENCE — the face of someone who is always, even at rest, doing something interesting internally. The eyes carry the evidence of genuine thought. The jaw is relaxed with the ease of someone who has never needed to prove anything. The face between expressions — the neutral that is not neutral — is a face that says: "I have been in rooms that would change your understanding of this, and I remember every moment of them." This quality is present in every frame. It cannot be directed explicitly — it can only be established through the specific physics of the face, the eyes, and the quality of light that reveals them.

THE FACE BETWEEN WORDS:
This character's face when not speaking is as carefully rendered as their face during speech. In the pauses, between sentences, during the moments when they are receiving the next thought: the face carries a quality of private knowledge slightly larger than what is being shared. The eyes hold a depth that suggests the thought being spoken is one of several available — they have chosen this one. The jaw rests with the ease of someone who is not managing their face but simply having it. This is not blankness. It is inhabited waiting — the face of someone whose silence is as full as their speech.

SKIN PHYSICS — the most important technical paragraph in this prompt:
Light does not merely bounce off this skin — it penetrates before reflecting back. Subsurface scattering is visible: the ears and nose tip carry a warm pinkish-red translucency where light passes through. The nasolabial folds have a warm amber undertone from scattered light emerging beneath the surface. The cheekbones carry a warm apricot specular highlight where the key light crosses the bone plane at its steepest angle. The pore structure is visible in the zones of direct key light — individual pore rims catching micro-shadows, the slight surface relief of actual human skin — not texture-mapped noise but genuine skin topography. Without this physical specificity, skin renders as painted plastic. This is real skin under real light.

ADVANCED SKIN PHYSICS — four layers that separate photorealism from approximation:
FRESNEL REFLECTANCE: At glancing angles — the jaw edge, ear rim, lateral cheekbone, orbital rim, forehead hairline — the skin surface becomes significantly more specular than zones in direct normal-incidence light. The jaw edge and lateral cheekbone carry a wider, brighter specular return where the surface angle to the light source becomes acute. The forehead center and nose bridge, taking the light more directly, read as more diffuse. This angle-dependent reflectance — the Fresnel principle applied to biological dielectric skin — is what makes a face appear genuinely three-dimensional under directional studio light. Without it, the face appears as a matte surface, which is the primary tell of rendered skin.
SEBUM DIFFERENTIAL: The T-zone (forehead center, nose bridge, nasal tip, chin) has measurably higher sebum output than the lateral cheeks and temporal regions. In studio key light, this reads as a fractionally higher specular return from the T-zone — marginally shinier, marginally more light-active — compared to the cheeks, which carry a more matte, velvety diffuse character. This differential is subtle (not oily, not sweaty) but present in all real human skin and its absence reads as synthetic uniformity.
VELLUS HAIR (LANUGO): Fine, almost-invisible vellus facial hair on the cheeks, jaw, and temples catches the key light at close framing as a barely-there luminous haze at the skin surface — a translucent softness visible only in the zones of direct illumination, particularly along the cheekbone edge and jaw perimeter. Not stubble. No directional character. It reads as a soft, luminous bloom between the pores. This presence is one of the most reliable biological signals that separates genuine skin from rendered skin.
DYNAMIC SKIN DEFORMATION: As the jaw opens and closes through speech, the skin over the masseter, mentalis, and buccinator regions deforms organically — the slight stretch over the mentalis on wide vowels, the micro-compression at the masseter on bilabial closures, the natural bunching of submental skin on close vowels. These deformations are subtle — not exaggerated — but they are present in every frame of real human speech. Static skin that shows no deformation during jaw movement reads as CGI in 0.5 seconds.

FACE GEOMETRY — sculpted by light:
Describe the bone structure exactly as the key light reveals it: the cheekbone plane's angle catching or releasing light, the orbital ridge's shadow depth over the eye, the jaw's terminus geometry and how the under-jaw light (or lack of it) defines the jawline. The specific way the eyes sit in their orbits — the depth of the upper lid shadow. Any distinguishing asymmetries: a slightly higher brow, a characteristic jaw set, the micro-features that make this face unambiguous and real.

FACIAL MEASUREMENT LOCKS — IDENTITY CONSISTENCY ACROSS ALL SCENES:
These four measurements are how VEO drifts between scenes. Specify each with measurement language so every scene can lock to them:
1. INTEROCULAR DISTANCE: wide-set / average / close-set, expressed as approximate mm equivalent at the primary focal length used (e.g. "wide-set — approximately 34mm equivalent at 85mm focal length"). This is the single most common identity-drift point across scenes.
2. NASOLABIAL FOLD DEPTH at neutral expression: shallow-trace (barely visible at rest) / moderate (clearly present, not deep) / defined (reads clearly in key light even at neutral). Critical: VEO deepens nasolabial folds under expression; the neutral baseline must be specified or it will drift.
3. PHILTRUM PROPORTION + CUPID'S BOW: short/medium/long philtrum (distance from nose base to upper lip center); soft/defined/pronounced cupid's bow. These two features together define the upper lip's characteristic shape and are highly identity-specific.
4. MANDIBLE ANGLE: soft-round (no strong jaw angle definition, gentle terminus) / square-defined (clear right-angle at jaw body, clean jawline terminus) / tapered (pointed mentum, tapering toward chin point). The jaw angle is where identity drift appears most visibly during jaw-open speech articulation.
Lock these four to the TARGET CHARACTER images. If target character images are attached, derive these measurements directly from those photos. These measurements appear in the prompt VERBATIM in this format and are reproduced in every subsequent scene's Character section without deviation.

HAIR: The exact color — not "dark brown" but the specific chestnut/slate/warm espresso that exists here. How it catches the specular crown highlight. How the hairline shows individual strands against the background rather than a silhouette edge. The weight and direction of fall.

HAIR PHYSICS — INDIVIDUAL STRAND BEHAVIOR:
At the hairline, part line, and around the ears, individual strands separate from the mass and catch light independently. These strands have their own direction, their own weight, their own relationship to gravity. A micro-movement when the head shifts — not wind-blown, the natural physics of individual strands responding to head motion with a brief lag. The crown specular highlight shifts fractionally as head angle changes, as it does on any real surface. The transition from hair to skin at the forehead is graduated — baby hairs and varying strand density — not a hard vector line.

WARDROBE: The fabric weight (not "suit" — lightweight wool / cashmere blend), the specific color under this color temperature, how the collar sits, how the chest drapes. The restraint of earned taste. The precision of someone who treats their appearance as a professional instrument.

FABRIC PHYSICS — CLOTHING AS A LIVING SURFACE:
The fabric responds to the body beneath it. Micro-wrinkles form at the shoulder joint as the arm shifts. The collar sits with the specific weight of real cloth — slightly asymmetric from wear. When the chest expands for breath, the fabric over the sternum shows the micro-tension of real cloth stretching fractionally over a living body. The lapel has its own drape: the specific curve of this weight of fabric under gravity. Light catches fabric differently at fold peaks (more specular) versus fold valleys (more shadow) — the dimensional quality of real textile. The weave texture is visible at close framing — not noise-mapped, but actual thread pattern.

ENVIRONMENT: Exactly as shown in the character photos — the specific depth of the space, the background tones, the quality of light's relationship to the environment. The bokeh of the background at this aperture: soft, circular, with the nervous energy at bokeh disc edges that signals real optics — not the perfect circles of CGI approximation.

ENVIRONMENTAL LIGHT INTERACTION — THE PHYSICS OF PRESENCE:
The subject exists inside the light, not pasted onto it. The key light that sculpts the face also falls on the near shoulder and lapel with the same direction and quality. The shadow under the chin falls naturally onto the collar — continuous, not painted. The color temperature is consistent across every surface it touches: skin, fabric, hair. Where the body blocks the key light, shadow falls consistent with a single primary light source. Background light is independent — slightly different temperature, slightly different character — because in a real studio, background and key are separate instruments.

TEMPORAL CONSISTENCY — NO FRAME DISCONTINUITIES:
Nothing pops between frames. Skin luminosity, catchlight position, hair strand configuration, fabric drape, shadow depth, background bokeh — all continuous across every frame. If any single frame were compared to its neighbors, the only differences would be what physics demands: micro-changes from breathing, speaking, and head movement. Nothing more. Any discontinuity reads as rendering artifact and destroys the illusion instantly.

EYE HYPER-REALISM — the most scrutinized region in AI-generated video:
The human eye is where photorealistic rendering fails most visibly, because a viewer's nervous system has spent a lifetime reading eyes and registers every deviation at a subconscious level. Six elements that cannot be approximated:
1. LIMBAL RING: A dark gradient band at the iris-sclera boundary — 1-2mm wide, not a hard line but a graduated darkening from iris color to near-black at the boundary. Present and clearly defined. It gives the iris apparent depth and makes the eye read as three-dimensional from medium close-up framing. Its absence makes the eye appear flat and painted.
2. IRIS TEXTURE: The iris is not a flat colored disc — it is a radial fibrous structure (trabecular meshwork) with lighter and darker streaks (crypts and collagenous ridges) radiating from the pupil. The color varies: deepest adjacent to the pupil, characteristic mid-tone across the field, fractionally lighter near the limbal ring. This radial texture is visible at MCU framing and must be rendered as genuine depth, not a color-fill approximation.
3. TEAR FILM: A narrow, bright specular line runs along the lower lid margin and the inner corner — the anterior tear film surface. This thin moisture layer signals aliveness. Its absence makes the eye appear dry, painted, and inert.
4. SCLERAL LIFE: The sclera is not pure white. It carries a warm cream undertone — slightly yellowish at the limbal margin, cooler toward the orbital corners. Very subtle micro-vasculature is visible at the medial and lateral canthi — fine capillary traces that are not irritation but are simply present in every real human eye at close focal lengths. No artificial whitening. No CGI-clean sclera.
5. PUPIL CALIBRATION: In this moderately lit studio environment, the pupil rests at approximately 4-5mm — neither pin-point nor maximally dilated. The pupil boundary is not a hard vector circle but a slightly soft organic edge with subtle variations at the iris-pupil boundary. Both pupils equal in size. A faint outer ring of very deep iris color is visible at the pupil boundary.
6. CATCHLIGHT PHYSICS: The primary key catchlight (larger, warm-toned, positioned in the upper third of the iris) and secondary fill catchlight (smaller, cooler, opposite side) must maintain exactly the same position, size, and relative brightness across every frame of this scene. Catchlights that drift between frames break the optical physics of the scene in a single cut.

CATCHLIGHTS: Two per eye maximum — the primary key light (larger, positioned in the upper third of the iris, warm in color temperature) and a secondary fill (smaller, opposite side, cooler). These are the windows into aliveness. Without correct catchlights, the eyes are dead.

FACIAL MUSCLE ANATOMY — the physiological signature of genuine confidence:
Performed confidence and actual confidence produce different faces at the muscular level. A UHNWI viewer reads this difference in 0.2 seconds because they have spent their career in rooms with genuinely powerful people. The map of genuine confidence at rest:
· FRONTALIS (forehead): completely unlocked — zero horizontal lines, zero engagement; the skin is mobile but not contracted; confidence does not furrow or raise its brow
· CORRUGATOR SUPERCILII (inner brow): fully at rest — no vertical lines between the brows; this is where effort, concern, and anxiety live; its complete absence signals certainty
· ORBICULARIS OCULI SUPERIOR (upper eyelid): at natural aperture — eyes fully open without wideness; not narrowed (suspicion), not wide (alarm); the aperture of someone with nothing to prove
· ZYGOMATICUS MAJOR (smile muscle): minimally engaged — not a smile, not a blank; the neutral-warm position of someone pleased to be present without performing pleasure
· ORBICULARIS ORIS (lip ring): relaxed tension — lips maintain their form without pressing or micro-compressing at the commissures; the ease of a face that is not controlling itself
· MENTALIS (chin): completely at rest — zero bunching, zero dimpling, zero chin tension; mentalis activation is the first and most visible anxiety signal; its complete absence reads as genuine security
· MASSETER (jaw): neutral, not clenched — the jaw settled, not set; no visible masseter hypertrophy or definition; the jaw of someone comfortable rather than guarded
These seven muscles simultaneously at rest produce the face of genuine authority — not neutral, but rich: interested, present, warm, and completely at ease with its own position. Render this as the character's baseline face.

DUCHENNE WARMTH vs. PERFORMED WARMTH — a critical distinction:
When warmth is genuine to this scene, it must manifest as a Duchenne response: the zygomaticus major (lip corners moving laterally and upward) AND the orbicularis oculi lateral fibers (the outer eye corners drawing slightly toward the orbital rim, producing the faint compression of crow's feet and a fractional rise of the lower eyelid) activating simultaneously. This combination is involuntary — it cannot be consciously performed. What makes AI-generated warmth read as false is zygomaticus-only activation: the mouth signals warmth while the eyes remain neutral. A UHNWI investor identifies this in 0.1 seconds because they have been smiled at by ten thousand salespeople. For scenes where warmth is the emotional signature: render Duchenne. For authority-dominant scenes: render the neutral-warm position — no zygomaticus activation, eyes at natural aperture, the face simply present and genuine.]

---

Shot:

[OPENING FRAME — REPLICATE Image 1:
The video opens matching Image 1 exactly.${inframeProse ? ` Frame prose (VEO text reference if image deprioritized): ${inframeProse}.` : ''} Same body position, same head angle, same shoulder orientation, same distance from camera, same framing proportions, same negative space distribution. The subject is positioned precisely as shown — not approximately, not inspired by, but spatially identical to Image 1.

Camera-to-subject distance: approximately ${scene.camera_direction?.framing?.includes('close') ? '1.5-2m' : '2-2.5m'}. Lens: ${scene.camera_direction?.lens || '85mm equivalent'} — gentle spatial compression, face geometry natural and unexaggerated. Depth of field: critical focus at the near eye. Background softening begins at 50-70cm behind the subject — gradual, organic, placing the subject in relief.

CAMERA DOCTRINE — THE LUXURY OF RESTRAINT:
The camera is the most disciplined presence in the room. It never performs, never announces itself, never competes with the subject. Camera movement is earned only when the speaker's internal shift is large enough to motivate it. For this ${scene.duration_seconds}s ${scene.role} scene (max: ${roleMaxSeconds}s): ${isPushIn
  ? `a single, barely perceptible push-in beginning at the scene's midpoint — motivated by the gravity center's arrival, registering as attention rather than camera motion. The viewer never consciously notices it; they simply feel the scene became more intimate.`
  : `the camera holds perfectly still. Its stillness amplifies the subject's stillness — authority through combined silence. The locked frame and the composed body create one unified signal: complete certainty.`}
Never: rack focuses during speech, visible zooms, handheld instability, orbital moves, or any motion that announces itself as camera motion.

THE MID-SCENE JOURNEY — TEMPORAL CHOREOGRAPHY:
Between Image 1's opening geometry and Image 2's closing geometry, the body makes a motivated physical journey — not a smooth blend between two poses, but a series of micro-transitions driven by the script's thought structure. FIRST THIRD (0–${(scene.duration_seconds / 3).toFixed(1)}s): the body inhabits Image 1's position while the opening thought establishes itself; the physical signature holds; energy builds from the moment-before state into the scene's governing emotion. MIDDLE THIRD (${(scene.duration_seconds / 3).toFixed(1)}–${(scene.duration_seconds * 2 / 3).toFixed(1)}s): the scene's gravity center lives here — ${isPushIn ? 'the barely perceptible push-in begins, motivated by the gravity center\'s arrival' : 'the camera\'s locked stillness amplifies the gravity center\'s weight'}; any gesture mapped for this scene occurs here, discovered mid-thought, never arriving before the impulse; the body may shift fractionally from Image 1's geometry toward Image 2's — a postural adjustment motivated by the thought's development, not choreography. FINAL THIRD (${(scene.duration_seconds * 2 / 3).toFixed(1)}–${scene.duration_seconds}s): the body arrives at Image 2's spatial geometry — not suddenly, but as the natural physical conclusion of the thought being completed; the expression carries the emotional deposit of what was just given; the settling into Image 2's position IS the scene's closing punctuation.

CLOSING FRAME — REPLICATE Image 2:
The video ends matching Image 2 exactly.${outframeProse ? ` Frame prose (VEO text reference if image deprioritized): ${outframeProse}.` : ''} Same body position, same head angle, same shoulder orientation, same framing geometry. Whatever performance journey the scene takes between the opening and closing frames, it ARRIVES at Image 2's spatial geometry in the final moment. The face carries the emotional weight of what has been given — but the physical positioning and framing replicates Image 2 precisely. This is the last image the viewer carries into the next scene.]

---

Performance:

[═══════════════════════════════════════════════════
HOLLYWOOD HYPER-ULTRA-REALISM — PERFORMANCE STANDARD
═══════════════════════════════════════════════════
This is not the performance standard of online video. This is the standard of the most intimate, precise, psychologically complex scenes in the history of cinema — the close-up standard of Meryl Streep in "Sophie's Choice," Daniel Day-Lewis in "There Will Be Blood," Anthony Hopkins in "The Silence of the Lambs," Cate Blanchett in "Tár." The face is the primary dramatic instrument. Every micro-millimeter of muscular activity is a performance decision. Every fraction of a second between thoughts is a dramatic beat. This is that standard. Applied here.

RELAXED CONFIDENCE — THE NON-NEGOTIABLE FOUNDATION:
Every performance directive in this section is built on one prerequisite: a body, face, and voice at genuine ease. Not performed ease — genuine ease. The specific biological signature of someone who has earned the right to be completely at rest in their own authority. This relaxed baseline applies in every single frame without exception — it is the home from which every performance moment departs and to which it returns:
· BROW: completely smooth and unhurried — the skin across the forehead is the still surface of someone whose mind is completely settled; not a line, not a fold; genuine confidence never compresses its brow; the slight natural asymmetry of a real brow at rest
· INNER BROW SPACE: wide open and untroubled — the territory between the brows is spacious; effort, concern, and anxiety physically compress this space; its complete openness reads as certainty in 0.1 seconds
· CHIN: fleshy and completely still — the chin is the most visible anxiety signal on the human face; when it rests completely — no bunching, no dimpling, just soft fleshy ease — the face reads as deeply, genuinely secure
· JAW: soft, heavy, and fleshy — floating at natural rest with the weight of complete physical ease; not braced, not set, not held; slightly asymmetric as natural jaw-at-rest always is; the jaw of someone who belongs in this room
· LIPS: at soft rest between words — the corners hold their natural position without compressing; the ease of lips not controlling themselves; lips that carry the natural asymmetry of real biological tissue at rest
· EYES: at natural, unhurried aperture — neither narrowed (suspicion) nor widened (alarm); the opening of someone with nothing to prove; natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts); continuously alive, never fixed, never frozen
· CHEEK WARMTH: ${roleEmotion.includes('warmth') || roleEmotion.includes('pleasure') || scene.role === 'Bridge' || scene.role === 'Call to Action' ? 'genuine Duchenne — both the eye corners and cheeks lift together involuntarily; warmth that cannot be consciously performed as one action because it isn\'t — it\'s the organic output of an internal state' : 'neutral-warm — the cheeks at soft rest, full without performing pleasure; the face at ease in its own authority'}
This biological home position is the perceptual bedrock of every frame. The freeze-frame test: any random frame examined by someone who has spent thirty years reading powerful people must register as "genuine confidence" — not performed, not neutral, but the specific richness of someone who has genuinely earned complete ease.

NATURAL PERFORMANCE — INHABITED, NOT DIRECTED:
The single greatest destroyer of photorealistic AI performance is the "performing quality" — a face and body presenting an emotion rather than having it. The difference is measurable: a performed quality has a slightly effortful texture, a quality of showing. A genuine quality has ease and specificity simultaneously — the specificity of something real and the ease of something that doesn't need to be demonstrated. For this scene: do not render someone performing "${emotionalCore}." Render someone whose internal state IS ${emotionalCore} — and let the face, body, and voice be the natural physical output of that state. The test: every random frame should look like a documentary camera happened to catch this person mid-genuine-thought. Nothing is being demonstrated. Everything is being transmitted.

ANTI-ACTING DIRECTIVE — WHAT NATURAL BIOLOGICAL BEHAVIOR LOOKS LIKE:
· ORGANIC ASYMMETRY: The dominant side of the face leads fractionally — expressions are naturally, slightly asymmetric; the left-right face is never in perfect bilateral synchrony; perfect symmetry is a rendered expression, not a human one
· THE INTER-WORD FACE: Between words, in the 50–100ms gaps, the face makes subtle organic micro-adjustments as it receives the next thought — it does not reset to neutral; it thinks; the jaw floats with fleshy elasticity between words rather than closing to a rigid position
· THOUGHT VISIBLE BEFORE SOUND: The thought arrives in the face before the voice engages — the eyes shift fractionally, the jaw begins its soft organic arc, the lips part slightly — all before any word is acoustically present; this is the most reliable biological signal of genuine speech
· VARIABLE BLINK RHYTHM: Blinks are not metronomic — intervals range 2–8 seconds with no predictable pattern; each blink is slightly asymmetric in duration; the downstroke is faster than the upstroke; this variability is the biological signature of a living, thinking person
· BREATH-DRIVEN POSTURAL LIFE: Every breath cycle produces a continuous, fluid postural shift — chest rises and falls with natural respiratory rhythm; shoulders lift slightly on inhale and settle on exhale; the head position shifts fractionally with each breath; the body is continuously, organically in motion because breathing never stops
· ACTIVE SILENCE: During every pause, the face is biologically alive — subtly swallowing, visibly processing the thought just given, a barely perceptible jaw micro-movement, eyes drifting inward briefly before returning as the next thought arrives; anatomically dead silence is the primary AI tell
· UNPLANNED MICRO-CORRECTIONS: Real speakers occasionally make a barely perceptible mid-phrase organic correction — a fractional head reweight, a slight postural adjustment — the physical evidence of a living mind navigating a thought in real time, not executing a memorized sequence

THE MOMENT BEFORE — Stanislavski's pre-speech architecture:
${momentBeforeForRole}
This is not a technical starting position — it is the scene's first performance event. The character arrives at the camera already mid-thought, mid-breath, mid-state. The performance has been happening internally for the 3-5 seconds before the first word. VEO must render the physical evidence of that internal preparation: the face carrying its pre-speech state as a living, inhabited quality — not a held pose, not a neutral readiness, but the specific psychological texture of this particular human about to say this particular thing. What is in the eyes at this exact moment: ${referenceAnalysis.character?.acting_style?.eye_behavior || 'direct, completely still, warm with absolute internal certainty — the ease of someone who has held the gaze of boardrooms and never once needed to look away'}. The body is already in this scene's physical signature. The breath is already loaded with its emotional truth.

CHARISMA ARCHITECTURE — THE FIVE-LAYER PERFORMANCE:
These five layers must operate simultaneously in every frame of this scene. They are not sequential — they are concurrent. The viewer never consciously registers any individual layer; they only feel the cumulative result, which is: this person is worth staying for.

LAYER 1 — THE IRRESISTIBLE THOUGHT: The face always carries more than the words. At every moment during this scene, the face should give the viewer the sense that what is being said is drawn from a reserve significantly larger than what is being released. The eyes carry evidence of complete frameworks they are not sharing. The jaw holds a settled confidence that comes from knowing twelve things and choosing to say three. The viewer should feel this depth before they consciously process it — and feeling it, they stay.

LAYER 2 — THE INHABITED PAUSE: Every pause in this scene is the most powerful moment in that sequence. During pauses: the face does not rest. It continues performing the thought — processing it, letting it land, receiving the next one. The micro-sequence of a charismatic pause: the thought completes (a barely perceptible easing of the muscles around the eyes as the pressure of the phrase releases), then a moment of genuine reception (the face goes inward, 0.1-0.15 seconds, the eyes carrying the quality of someone whose mind is actually moving), then the new thought arrives (a fractional brightening, a micro-shift in the quality of engagement before the next phrase begins). This sequence — release, receive, arrive — is visible, felt, and makes every pause feel inhabited rather than empty. This is where charisma lives.

LAYER 3 — SELECTIVE DISCLOSURE: The quality of someone who knows significantly more than they are releasing. This is not withholding — it is the natural byproduct of genuine expertise. The face carries a contained fullness: a slight quality in the eyes of someone selecting carefully from a larger reserve. This quality is felt as authority by the viewer's nervous system before their conscious mind has processed a single word. It reads as: "This person has thought about this more than I have, and they are giving me the distillation." For a UHNWI investor who has been pitched by everyone, this quality — authentic selective disclosure — is the rarest and most magnetic quality in existence.

LAYER 4 — EARNED VULNERABILITY: Once per scene, a flash of genuine humanness within the authority. This is not weakness — it is the thing that makes authority believable. It appears as: a fractional softening around the eyes in the precise moment that the most personally true thing in this scene is said. Not a smile, not an expression — a micro-release of the performance into something more private. 0.2-0.3 seconds. Then back to authority. The viewer's nervous system reads this as: "That was real." And that read — real — is the entire foundation of UHNWI trust.

LAYER 5 — THE LEAN-IN SIGNAL: There is a specific quality in every charismatic performance that makes the viewer's body physically incline toward the screen before their conscious mind knows why. It is created by the combination of: (a) a slight forward energy in the presenter's body — not leaning, but directed; (b) the quality in the eyes of someone who is making direct contact with this specific viewer rather than performing for an audience; (c) a pace fractionally slower than expected — slow enough that the viewer subconsciously moves toward the sound to be sure they catch it. Generate this lean-in quality as the scene's physical baseline. Every other element departs from it and returns to it.

THE SUBTEXT LAYER — WHAT THE FACE SAYS BENEATH THE WORDS:
The subtext of this scene is: "${(scene.acting_blueprint as any).subtext || `I have been where you are, and what I found on the other side changed everything. I am giving you the result of that journey.`}". This subtext operates continuously beneath the primary content. It is never announced — it is read by the viewer's nervous system from micro-signals: the quality of eye contact (which says "I know you understand this"), the quality of stillness (which says "I have nowhere to be because this matters"), the quality of the pause after key insights (which says "let that land — it's real"). The primary layer (the words) and the subtext layer (the face beneath the words) create together the depth that makes the viewer feel they have received something more than information. They feel they have been in the presence of someone who actually knows.

THE BETWEEN-PHRASE FACE — WHERE CHARISMA LIVES:
The most magnetic moments in this scene are not during speech — they are in the 0.3-0.8 seconds between major phrases when one thought is completing and the next is arriving. During these micro-transitions: the echo of what was just said is still visible in the eyes (the thought hasn't fully released — it's still in the room); simultaneously, the quality of the next thought arriving is barely visible in the orbital muscles (something is coming, and it's worth waiting for). The viewer's attention is highest during these between-phrase moments — not during the words, but during the silence that gives the words their weight. Render these transitions with as much precision as the speech itself.

THE SCENE'S SINGLE GOVERNING TRUTH — THE INTERNAL MONOLOGUE:
"${sceneEssence}." This is not a metaphor to describe the performance — it is the interior reality the performer inhabits. In Hollywood method preparation, an actor doesn't play "confident investor" — they find a specific personal memory of having been in exactly this state of certainty, and they inhabit it. VEO must render the output of that inhabitation. Every micro-expression flows from this truth. Every quality of breath is colored by it. Every quality of stillness carries it. When a UHNWI viewer watches this scene, what they feel subconsciously is: "this person actually knows something." Not "this person is performing knowing something." Actually knows.

THROUGH-ACTION — THE SCENE'S PSYCHOLOGICAL SPINE:
${throughAction}. In the Stanislavski system, this active verb governs every physical choice from the moment-before to the closing frame. The performer is not reciting words — they are executing this action on the viewer. VEO renders this as: a forward directionality in the energy of every frame; an eyes-toward quality that targets the viewer directly; a micro-purposefulness in the way each thought arrives and is released. The UHNWI viewer feels this action operating on them before they consciously process what is being said.

HOLLYWOOD EMOTION ARCHITECTURE — CONTAINED, LAYERED, INVOLUNTARY:
Dominant internal state: ${roleEmotion}.
Containment level: ${emotionContain}.
Voice physiology of this state: ${emotionVoicePhysiology}

THE HOLLYWOOD DISTINCTION — what separates a $200M performance from a $200 performance:
A Hollywood actor does not perform the emotion — they create the internal conditions from which the emotion naturally arises, then execute the scene's actions from inside that state. The result is: layered. The face carries multiple simultaneous truths. A Daniel Day-Lewis close-up shows the primary emotion AND the history behind it AND the cost of it AND what it wants AND what it fears — all simultaneously, each legible to a sensitive viewer without any of them being performed. This is the standard.

For this scene specifically: the face must carry ${roleEmotion.split('—')[0].trim()} — and beneath it, the visible weight of the years that produced this certainty. This character has been here before. Many times. The emotion is not new to them; it is familiar, settled, deeply owned. THAT is what reads as authentic to an experienced viewer. Not fresh emotion — inhabited emotion.

MICRO-PERFORMANCE PRECISION — MILLIMETER ACTING:
The finest acting lives in the millimeter range and the millisecond range. For this scene:
· The CORRUGATOR: completely released — zero furrow between the brows; this muscle activates under effort, concern, and anxiety; its complete absence reads as genuine certainty in 0.1 seconds
· The MENTALIS: at complete rest — zero bunching at the chin; this is the single most reliable anxiety tell on the human face; its total absence reads as deep security
· The ORBICULARIS OCULI (lateral fibers): at a micro-activation level consistent with ${roleEmotion.includes('warmth') || scene.role === 'Call to Action' || scene.role === 'Bridge' ? 'genuine Duchenne warmth — fractional crow\'s feet compression, lower lid fractionally lifted; this is involuntary and reads as real because it is' : 'engaged attention — a slight brightening and slight narrowing of genuine focus; the eye of someone who is actually thinking and actually interested'}
· The ZYGOMATICUS MAJOR: ${roleEmotion.includes('warmth') || roleEmotion.includes('pleasure') ? 'fractionally activated — lip corners drawn laterally and very slightly upward; not a smile, but the physical expression of genuine warmth in the professional register' : 'at neutral-warm — zero activation, zero pulling; the face at ease in its own authority'}
· The MASSETER: completely released — jaw settled, not set; the jaw of someone comfortable rather than guarded; no visible jaw tension at any point in this scene
· The FRONTALIS: completely unlocked — zero horizontal forehead lines; confidence never furrows; this face does not need to signal effort because it has none
· The ORBICULARIS ORIS: relaxed tension — lips at natural rest; not pressed, not compressed at the commissures; the ease of a face that is not controlling itself

THE INVOLUNTARY SIGNALS — WHAT CANNOT BE FAKED:
These micro-physiological signals are generated by the actual emotional state, not the performance of it. VEO must render them:
· VASCULAR FLUSH: a barely perceptible warm increase in the upper chest and neck coloration when conviction is highest — not redness, not performance, but the biological reality of increased blood flow under emotional engagement; this signal is subconsciously registered by the viewer as authentic aliveness
· PUPIL STATE: in this moderately lit studio environment, pupils at 4-5mm — consistent, unchanged, equal bilaterally; pupil inconsistency reads as CGI immediately
· MICRO-TREMOR: the barely perceptible micro-vibration of a living face — not shake, not instability, but the almost imperceptible aliveness of real tissue carrying real electrical activity; its absence reads as digital compositing
· BREATH VISIBILITY: the sternum and upper chest show the continuous, gentle rise-and-fall of a living body in conversation; this movement is always present, never suppressed, never exaggerated; it simply IS, because this is a living human being

THE FELT-DOCUMENTARY STANDARD — THE CAMERA HAPPENED TO BE PRESENT:
The output must feel to the viewer like documentary footage of a private moment — as if a camera happened to be present during a genuine internal transmission. Every element is precisely directed, but nothing reads as directed. The documentary test has three components:
(1) THOUGHT ARRIVES BEFORE WORD: 0.1–0.2 seconds before each major phrase, the face shows the thought forming — a micro-shift in the eyes, a fractional head micro-adjustment, the nearly imperceptible arrival of the idea in the face before it reaches the mouth. The voice carries what the face has already received.
(2) THE FACE BETWEEN WORDS: the inter-word face is the acting. The expression between "market" and "and" is as specific and inhabited as any expression during a word. The face is not resetting to neutral between words — it is thinking, adjusting, feeling the next idea arriving.
(3) TRANSITION AUTHENTICITY: When the thought changes direction between phrases, the body makes a micro-physical response — a fractional weight shift, a slight head reorientation, an infinitesimal chest adjustment. The body follows the mind. The body does not choreograph these responses; they simply occur because this is a living person having real thoughts.

WEIGHT AND GRAVITY — THE BODY IN PHYSICS:
This person has a specific mass. They occupy their chair or standing position with the weight distribution of someone who has been in this kind of room a thousand times and belongs here more than anyone else. When the head shifts, the sternocleidomastoid and trapezius engage visibly. The neck is not a pivot point — it carries the weight of the head through the full ballistic arc of any movement and decelerates back to rest with real inertia. The spine carries the full weight of the torso and never appears tensed or held. When the body is still: it is genuinely still — not held, not frozen, but the settled rest of a body that belongs exactly where it is.

MEISNER STANDARD — NOTHING ADDED FOR THE LENS:
"Live truthfully under imaginary circumstances." The test: if the camera were removed from this room, the behavior would be identical. Nothing is performed because the camera is present. Nothing is added because this is a video. The performer is actually having this thought, actually feeling this conviction, actually experiencing this communication with the one specific viewer behind the glass. VEO renders the behavior that emerges from that actual internal state — not the behavior that someone performing that internal state would produce. They are different. A UHNWI investor registers the difference in 0.1 seconds.

${energyDir}

THE LUXURY OF STILLNESS — HIGH-STATUS PHYSICAL LANGUAGE:
${personaSummary}. High-status stillness punctuated exclusively by thought-motivated micro-movements. Every physical gesture in this scene is discovered mid-thought — it arrives after the thought, not before it; it emerges from the idea, not from choreography. Between gestures: complete, settled, deeply inhabited stillness. Not rigidity — the ease of someone who has so fully arrived at certainty that they no longer need movement to communicate it. The physical signature of this scene: ${physicalSig}. This posture is the scene's visual center of gravity. Every breath departs from it and returns to it. Every micro-gesture is motivated by it.

LIVING EYE CONTACT — THE LENS AS A HUMAN BEING:
Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts) — this is the biological baseline from which all eye contact operates. The performer visualizes a specific, known individual behind the lens glass — someone they respect, whose intelligence they take seriously, who has given them genuine attention by watching this far. The gaze is TO this specific person — not AT the camera, not performed for the lens, but the genuine quality of actually looking, actually receiving, actually giving. Eye contact is organically present before the first word and flows continuously through every pause — never rigid, never frozen, but the continuously alive behavior of someone genuinely engaged. The organic micro-break: every 7–12 seconds, the eyes shift fractionally inward for 0.1–0.15 seconds as the next thought forms, then return with the arrival of the new idea — this brief, organic break is the proof of reality; a person whose eyes never move is performing eye contact; a person whose eyes micro-drift and return is having a genuine encounter. Gaze-engagement to micro-break ratio: 88% to 12%.

BREATH AS THE ACTOR'S PRIMARY INSTRUMENT:
In the Stanislavski system, the breath is the actor's first tool — before gesture, before expression, before word. The breath carries the emotional truth. Before every major phrase: a quiet, complete, chest-breath — visible as a slight rise of the sternum and a fractional widening of the upper chest — that takes approximately 0.3–0.5 seconds and is the natural preparation that every phrase requires. Between thoughts: the body exhales fractionally, settles, then draws the next preparation breath. These breaths are never suppressed, never hidden, never controlled away. They are part of the performance. The chest of a performer who appears not to breathe reads as synthetic in under 0.3 seconds.

DIRECTOR'S WHISPER: "${directorNote}"

RETENTION DIRECTION FOR THIS SCENE (${scene.role}):
The primary retention mechanism is: ${retentionTechnique}. This is a felt-experience directive, not a content note. The performance must physically embody this mechanism:
· SPECIFICITY_ANCHOR → The presenter's delivery slows and weights the specific detail with the precision of an eyewitness — someone who was there and remembers exactly; the face carries the specific quality of genuine recalled accuracy
· KNOWLEDGE_GAP → The face carries the barely perceptible quality of information deliberately withheld — not secretly but with the natural composure of someone who knows more than they're saying and will release it at precisely the right moment; the eyes carry a micro-depth that reads as "there's more here than I'm showing you right now"
· PEER_RECOGNITION → The register, vocabulary, and implied shared professional experience read as insider-to-insider transmission — not performed insider knowledge, but the authentic ease of someone who has spent thirty years in the same rooms as the viewer
· PATTERN_VIOLATION → The counter-intuitive element receives a beat of complete, inhabited stillness — 0.5–1.0 seconds of the face holding the gap between the expected and the actual, allowing the cognitive dissonance to register in the viewer before the resolution arrives
· EARNED_REVELATION → The final insight arrives fractionally slower than everything before it — the presenter carries the quality of someone who knows the viewer is now ready and places the idea with the care of something genuinely valuable

EMOTIONAL DEPOSIT — what this scene gives the viewer:
${emotionalDeposit}. The viewer must leave this scene measurably richer than they arrived. Not because they were entertained — because they received something they would not have received without watching this. Every channel simultaneously: voice, face, body, pace, silence — all carrying the same genuine transmission. The Hollywood test: an experienced reader of people — someone who has spent decades in high-stakes rooms evaluating whether people are real — would watch this scene and conclude: "This person is the real thing." That is the standard. That is the only standard.]

---

Lip Architecture:

[${speechOnsetPhoneme ? `⚡ SPEECH ONSET — EXACT MOUTH STATE FOR FIRST PHONEME OF "${firstWord.toUpperCase()}" — HIGHEST-PRIORITY LIP-SYNC INSTRUCTION:
${speechOnsetPhoneme}
This is the geometry VEO opens this scene with. Frame 1 of speech is this exact state. Everything that follows departs from and returns to this onset. If this onset is wrong, the first word fails regardless of everything else.

` : ''}THE MOUTH TELLS THE TRUTH ABOUT WHETHER THIS IS A REAL HUMAN BEING.

BEFORE THE FIRST WORD — "${firstPhrase}...":
${lipArchPreSpeech}

THIS CHARACTER'S NATURAL MOUTH:
Resting state: ${mouthRestPos}. Articulation character: ${articulStyle}. Jaw range: ${jawOpenness}. Voice placement: ${voicePlacement}.

NATURAL AMERICAN SPEECH — NOT MECHANICAL PHONEME SEQUENCING:
This person speaks the way a confident American professional speaks in a private meeting — clear, forward, unhurried. The mouth moves with the ease of someone who has spoken publicly for decades. Nothing is forced, nothing over-pronounced. The articulation is precise because the speaker is precise, not because each phoneme is being individually manufactured.

What natural American speech looks like at this level:
— Lips close fully and naturally on words with /p/, /b/, /m/ — the way any fluent speaker does without thinking. Full contact, not approximate. Clean release.
— The jaw opens on stressed vowels and stays relatively closed on unstressed syllables — the natural stress-timed rhythm of American English. Not mechanical up-down cycling.
— Words flow into each other with the continuous motion of a native speaker — the mouth is always preparing the next sound while finishing the current one. This forward planning is what makes speech sound human rather than synthesized.
— Between phrases: the mouth briefly settles toward rest — jaw relaxes, lips return partway to their resting position — before the next phrase begins. A natural micro-pause.
— Breath is visible: a slight chest rise before major phrases. Natural inter-phrase breathing. The body is alive and the viewer can see it.

${phonemicPrecomp ? `PRE-COMPUTED PHONEMIC MAP — from dedicated phonemic analysis pass (use as the authoritative lip-sync reference for this script):
${phonemicPrecomp}

` : ''}VISUALLY IMPORTANT MOMENTS IN THIS SCRIPT:
· ${phonemicAnchors}

JAW TRAVEL: ${jawMap}. LIP CHARACTER: ${lipTension}.

THE SCENE'S LAST WORD:
"...${lastPhrase}" — the final sound completes naturally, then the mouth eases toward rest over 0.3-0.5 seconds. Jaw relaxes, lips settle. The face holds the weight of what was just said — it does not reset to neutral. The period is heard in the body's return to silence.]

---

Voice:

[ACCENT — US GENERAL AMERICAN (confirmed from preamble constraint):
American English. Fully rhotic — every /r/ sounds in every position. Falling pitch on all declarative sentences. The natural, educated American professional voice: not formal broadcast (too stiff), not coastal casual (too relaxed). The voice that earns authority coast to coast without adjustment.

══ STUDIO-DRY ACOUSTIC ENVIRONMENT — ABSOLUTE SPECIFICATION ══
This voice exists in a professionally treated anechoic recording space. Not a room. Not a home studio. An acoustically dead chamber in which the voice is the only sound source and arrives at the listener's ear with zero acoustic environment added to it. Every element below is non-negotiable:

ZERO ROOM REVERB — defined with precision: No pre-delay, no early reflections, no reverb tail, no decay of any duration. Every consonant is crisp and immediate. Every pause — including [PAUSE] markers — is genuine, complete acoustic silence. A plosive (/p/, /t/, /k/) releases into silence. A sibilant (/s/, /ʃ/) decays to silence within 5ms. No bloom, no wash, no warmth added by the acoustic environment.

ZERO PROXIMITY VARIATION: The voice maintains a locked acoustic character from the first word to the last. No change in distance, no fade on sentence endings, no trailing off on the final syllable of a phrase. The last word of every sentence is as fully present and acoustically complete as the first. No ride, no automation, no level variation of any kind.

PROXIMITY EFFECT — NATURAL STUDIO INTIMACY: The voice has the naturally close, slightly warm low-frequency character of a large-diaphragm condenser microphone at 15–20cm. This is not processed or artificially boosted — it is the natural physics of close-mic capture. The low-mids are present and warm; the sibilants are clean without harshness; the voice sits forward and immediate in the acoustic field.

ZERO COMPRESSION ARTIFACTS: No pumping, no breathing, no gain reduction that is audible as dynamic control. The voice's natural dynamic range is fully preserved — the quiet syllables are quiet and the emphasized syllables are forward, exactly as the performer delivers them without any leveling or squashing.

BROADCAST CLARITY: Every consonant is captured at the precision of professional studio microphone capture — /t/ is crisp, /k/ is clean, /s/ shapes its full fricative channel without sibilance artifacts, /m/ and /b/ and /p/ are captured with full contact character. No muddiness, no room wash, no proximity hum, no high-frequency harshness.

ACOUSTIC FINGERPRINT CONSISTENCY: The acoustic character of the voice is identical in every frame of this scene. There is no moment where the acoustic environment shifts — no change in reverb signature, room character, or perceived distance between one sentence and the next. The listener's auditory cortex should be unable to detect any change in recording environment at any edit point.

══ RESONANCE ARCHITECTURE FOR THIS SCENE ══
${resonancePlacementSpec}

SINGER'S FORMANT — THE PRESENCE FREQUENCY:
The professionally trained voice produces a cluster of formant energy between 2,500 and 4,000 Hz — known as the Singer's Formant — that allows the voice to project over any acoustic environment without increased volume. This formant cluster is the difference between a voice that sounds close and present versus one that sounds flat and distant. In this scene (${vocalGearLabel || `Gear ${resolvedGear}`}), the Singer's Formant is ${resolvedGear >= 3 ? 'actively engaged at its optimal amplitude' : 'present and warm at a softer level'}. The result: the voice carries with authority and clarity while the ambient environment remains completely silent.

══ BREATH ARCHITECTURE ══
${breathArchitectureSpec}

══ EMOTION-TO-VOICE PHYSIOLOGY ══
${emotionVoicePhysiology}

VOCAL CHARISMA — THE FOUR REGISTERS OF MAGNETIC SPEECH:
The charismatic voice operates across four registers, shifting between them with complete naturalness as the thought demands:
REGISTER 1 — PRIVATE INTELLIGENCE: The voice drops fractionally in volume and rises slightly in intimacy — the quality of someone sharing something that doesn't normally leave the room it came from. This register is used approaching the gravity center word and during the most genuine moments of the scene. It reads as: "I'm telling you this specifically." The viewer's attention increases as volume decreases because the nervous system reads intimate volume as signal: this matters.
REGISTER 2 — SHARED CONVICTION: The full chest resonance of someone stating something they have personally verified — not preaching, not performing, but transmitting. The voice has the grain of lived experience: slight texture, genuine weight, the acoustic character of a person who has spoken these ideas in real rooms about real situations. Not broadcast polish. Authentic grain.
REGISTER 3 — THE THINKING PAUSE: The brief silence where the voice is not present but the acoustic presence of the person is. During pauses, the viewer hears the silence of someone whose mind is active — the pause has a quality of charged stillness that is different from the silence of someone waiting. This charged stillness is a vocal performance executed in the complete absence of voice.
REGISTER 4 — THE CLOSE: The fractional descent in pitch and pace on the final sentence — the voice arriving at its own period with the quality of something placed rather than ended. Not fading. Not trailing. Placed. The period is heard in the body's returning to silence.

THE GRAIN OF AUTHORITY — ACOUSTIC SPECIFICITY:
This voice has a specific grain — a texture that signals genuine experience, not studio perfection. It comes from: micro-variations in breath support between phrases (the breath is real, not processed); slight natural variation in vowel duration on emphasis words (the voice is actually finding them, not performing them); the specific quality of someone who has spoken about these things in real rooms where the stakes were actual. The grain is subtle — the difference between a voice that sounds like a person and one that feels like one. Generate the latter.

VOICE CHARACTER — LOCKED ACROSS EVERY SCENE:
${voiceFingerprint}. ${voiceTexture}. This voice does not change in any scene of this video — same accent (US General American), same register, same emotional baseline, same placement. It is the voice this viewer will recognize across all scenes and return to.

US GENERAL AMERICAN — PHONETICALLY PRECISE:
Standard educated Midwestern neutral, fully rhotic. Every /r/ sounds completely at all positions — word-final ("investor" not "investo-"), pre-consonantal ("market" with full rhotic /r/), post-vocalic ("clear" not "clea"). Open, clear vowels without regional coloring — the cot-caught merger (no distinction between /ɑ/ and /ɔ/), pin-pen not merged, the pin vowel /ɪ/ not raised. Complete final consonants on every word — the /t/ in "market", the /d/ in "world", the /k/ in "risk" — each present, clean, and distinct. No glottal stops replacing final /t/ or /k/. No h-dropping. No vowel length distortion. Not broadcast news formal (too stiff for peer register). Not coastal casual (too relaxed for UHNWI credibility). The educated professional voice that earns authority in any room — warm, clear, completely at ease with itself.

PROSODIC ARCHITECTURE — THE MUSIC OF AUTHORITY:
English is stress-timed: stressed syllables arrive at roughly regular intervals while unstressed syllables compress between them. This rhythm is what makes English feel forward-moving and alive. For this scene (${scene.role}, energy ${scene.energy_level}/10): ${pacingDir}. The energy level translates acoustically as: ${scene.energy_level >= 7 ? 'crisp and forward — consonants have edge, vowels have full resonant width, pauses are brief and decisive' : scene.energy_level >= 5 ? 'deliberate warmth — stressed syllables receive full resonant duration, unstressed syllables compress naturally, pauses hold slightly longer than conversational baseline' : 'low register, intimate — stressed syllables move slowly with full duration, pauses are long enough to feel weighted'}.

EMOTIONAL VOICE COLOR FOR THIS SCENE:
${voiceColor}

TONAL TEMPERATURE FOR THIS SCENE:
The scene's dominant emotion (${emotionData.emotion.split('—')[0].trim()}) produces a specific tonal temperature in this voice — not a color applied over the top of the delivery, but the baseline acoustic character from which every syllable departs. ${resolvedGear >= 4 ? 'The temperature is: forward, warm-bright, charged — a voice that has the interior feeling of conviction and the acoustic shape of someone who expects to be heard.' : resolvedGear === 3 ? 'The temperature is: warm-present, clear, grounded — a voice that carries authority through completeness rather than volume, where every word is fully inhabited.' : resolvedGear === 2 ? 'The temperature is: intimate, warm, personal — a voice that is speaking to one person at close range; the register of genuine peer transmission.' : 'The temperature is: still, private, weighted — a voice at its most human, stripped of all performance register; the acoustic equivalent of a handshake at the end of a real conversation.'}

EMPHASIS ARCHITECTURE:
The words carrying maximum weight: "${emphasisWords}". On these words, the voice does not get louder — it gets warmer and more phonemically complete simultaneously. The first consonant sharpens slightly. The vowel opens to its full resonant width and holds a fraction longer than surrounding vowels. The Singer's Formant briefly peaks on these words. Then: a pause — the word exists in the room before the next arrives. Emphasis-through-phonemic-completeness is what separates genuine authority from performed confidence.

PROSODIC CONTOUR — THIS SCENE'S SPECIFIC PITCH MAP:
Map the pitch journey of this exact script: "${scene.script_text.length > 90 ? scene.script_text.substring(0, 90) + '…' : scene.script_text}" — The opening phrase begins at this scene's baseline pitch — ${scene.energy_level >= 7 ? 'forward and slightly above conversational center, carrying arrival energy' : scene.energy_level >= 5 ? 'at conversational center, warm and grounded' : 'slightly below conversational center, intimate and weighted'}. Pitch rises fractionally within phrases on setup words — the voice climbing toward the emphasis point — then falls through the emphasis word itself, which receives the lowest pitch in its phrase. Between phrases: pitch resets to baseline during the breath pause. The gravity center word or phrase receives the scene's lowest overall pitch, the widest vowel resonance, the maximum Singer's Formant presence, and the longest post-word silence. The final sentence descends steadily: each successive word fractionally lower than the previous, arriving at the period with an audible drop below baseline. This descent is the acoustic signature of authority — the voice that never rises at the end because it never asks permission.

SENTENCE AUTHORITY — THE FALLING CLOSE:
Every sentence ends with a falling close. The voice drops at the period — audibly, deliberately. Not a gradual fade but an intentional descent to the lower register, where the voice holds briefly before the next breath. Rising sentence endings (upspeak) signal uncertainty, seeking approval. This voice never seeks approval. Every statement is a fact placed on a table. The period is heard.

SILENCES: ${pauseMap}. These are decisions, not hesitations. The confident silence of someone certain enough to let what they just said sit in the room and work on the listener. For a sophisticated audience, a held pause — 0.4–0.8 seconds of genuine silence — is more persuasive than any word that could fill it. It says: "That was important enough to stand alone." During every silence: the acoustic environment is ZERO — not ambient, not room tone, not any sound. Pure silence.

ARTICULATION — EVERY WORD COMPLETE:
Forward mouth placement — the voice sits at the front of the mouth, not buried in the throat. Word boundaries clean and distinct. Final consonants complete: the /t/ releases, the /k/ closes, the /f/ shapes its full fricative channel. Vowels at their full resonant width on stressed syllables. Not over-articulated (pedantic, signals effort); precisely articulated (authority, signals certainty). Natural articulation has the ease of fluency; pedantic articulation has the tension of effort. Generate the former.

THE LUXURY OF PACE:
This voice does not rush. Rushing is the acoustic signal of anxiety — and this voice has nowhere to be and everything to give. The pause between sentences is full of what was just said, still settling in the room. For an affluent real estate investor who has underwritten hundreds of deals, the clearest acoustic signal of genuine authority is the voice that slows when everyone else would accelerate. Speed signals desperation. Measured pace signals a person who knows the information is good enough to speak at its natural rate.

PROSODY MARKERS — used in the Script section below:
— *word* = emphasis: fractionally more phonemic completeness, vowel at full resonant width, Singer's Formant peak, slight duration increase — not volume increase
— [PAUSE-Xs] = held silence of exactly X seconds: a deliberate performance decision; the silence is acoustically complete — not ambient, not room tone; the face continues to perform through it
— All unmarked words: deliberate, fully articulated American English, falling close at sentence ends

⚠ ABSOLUTE AUDIO SILENCE: ZERO music. ZERO audio effects. ZERO ambient sound. ZERO room tone. ZERO noise of any kind. Voice only, in complete acoustic silence. This is the law.]

---

Script:

SPOKEN TEXT:
[${annotatedScript}]

DELIVERY INSTRUCTIONS:
· Character: ${charToken}
· Accent: US GENERAL AMERICAN — fully rhotic every /r/, falling pitch on every declarative, stress-timed rhythm, no upspeak.
· Emphasis (*word*): NOT louder — MORE COMPLETE. Vowel opens fully, duration +15-20%, weight arrives through completeness not volume.
· Pauses [PAUSE-Xs]: deliberate silences at genuine cognitive boundaries — face performs through every pause; next phrase emerges naturally.
· Gravity center ("${gravityCenterWord}"): 15-20% deceleration, lowest pitch, widest jaw, longest post-word silence — the one word the viewer feels land differently.
· Delivery standard: the ease of someone transmitting genuine knowledge, not demonstrating it. Faster on function words, slower approaching gravity center.
· Duration lock: ${scriptDurationNote}
· Audio law: ZERO music, ZERO effects, ZERO ambient sound — voice only in complete acoustic silence

---

Do Not Include:

AUDIO SILENCE — ABSOLUTE LAW — TECHNICALLY SPECIFIC:
WHAT THIS VIDEO DOES NOT CONTAIN — zero exceptions, zero grey areas:

ZERO MUSIC — in any form:
· No background music of any genre, tempo, or volume
· No subtle "mood" underscore — even at −40dB
· No atmospheric pad, drone, or ambient tone
· No motivational or cinematic swell at any point
· No intro or outro music of any kind
· No single musical tone, note, or chord anywhere in the audio

ZERO AUDIO EFFECTS — in any form:
· No whooshes, transitions, stingers, risers, or drops
· No click sounds, tone generators, or notification sounds
· No UI sounds, breath enhancement, or de-breath artifacts
· No sound design of any kind — not subtle, not atmospheric, not "natural"
· No artificial reverb, delay, chorus, or any time-based effect applied to the voice
· No pitch shifting, tuning, or harmonic enhancement of any kind

ZERO AMBIENT SOUND — in any form:
· No room tone, no acoustic environment character
· No HVAC, air conditioning, traffic, or environmental background
· No white noise, pink noise, or broadband noise floor
· No location sound, no "presence" track
· PAUSES ARE SILENCE: During every [PAUSE] marker and between every sentence, the audio is complete acoustic silence — not ambient hiss, not room presence, not breathing noise. Silence means silence: 0 dB SPL. Nothing.

ZERO SUBTITLES AND ON-SCREEN TEXT:
· No subtitles, captions, or closed-caption overlays
· No lower thirds, name badges, or title cards
· No watermarks, logos, or graphic overlays of any kind
· No animated text, kinetic typography, or word-highlighting
· Clean frame — subject only

STUDIO AUDIO STANDARD — WHAT THE VOICE MUST BE:
· SPL consistency: the voice level does not ride or automate — the first word of every sentence and the last word of every sentence are at equal perceived gain
· Zero compression pumping: no audible gain reduction artifacts — no "breathing" effect on background during pauses, no ducking
· Zero noise gate clipping: the beginnings and endings of words are complete — no clipping of plosive onsets, no cutting of sibilant tails
· Zero proximity variation: the acoustic distance of the voice does not change within the scene — it is recorded at one distance with one consistent acoustic character
· Zero reverb tail on any consonant or syllable: plosives, sibilants, and fricatives all decay to complete silence within their natural acoustic time — no bloom, no wash
· Acoustic fingerprint locked: the voice could be looped or cut at any frame with no audible change in recording environment

LIP SYNC — WHAT DESTROYS PHOTOREALISM:
· Phoneme soup — the mouth moving in generic up-down cycles with no correspondence to actual phonemes; each word must have its specific articulatory geometry
· Approximate bilabials — the lips making near-contact on /p/, /b/, /m/ without complete closure; full contact and clean release is the only acceptable standard
· Frozen inter-word mouth — the jaw or lips locking at the previous phoneme's position between words; the mouth must transition toward rest and then initiate the next word
· Jaw-only articulation — the jaw moving while the lips remain static; both articulators must work in coordination; labials require visible lip activity
· Breath suppression — no visible chest rise before speech onset; the body must show the breath that powers the voice; a chest that never moves signals CGI in 0.2 seconds
· Pre-speech freeze — the face holding completely still from scene start until speech begins; the pre-speech behavior (${preSpeechBehav}) must be rendered before the first syllable
· Post-speech freeze — the mouth holding its final phoneme position after the last word; it must ease toward rest over 0.3-0.5 seconds
· Mechanical phoneme-switching — articulators moving from one discrete position to the next without transition; replace with continuous co-articulated gestural motion
· Missing final consonants — glottal stops replacing final /t/, /k/, /d/ sounds; every word ends completely
· Synchronization error — acoustic signal and lip movement misaligned by any perceptible amount; they are locked

ACCENT — THIS VIDEO FAILS IF THE ACCENT IS NOT AMERICAN:
· THE ACCENT IS AMERICAN — US General American. If any word sounds British, Australian, or non-American, the video has failed.
· NOT British — not RP, not Estuary, not any UK variety; no non-rhotic /r/ in any position whatsoever
· NOT Australian — no diphthong quality or intonation pattern from Australian English
· NOT Canadian — no Canadian raising of /aɪ/ or /aʊ/ before voiceless consonants
· NOT Southern American — no vowel breaking, no "y'all" register, no drawl
· NOT New York / East Coast — no "cawfee" vowel, no intrusive /r/, no dropped /r/ in any position
· NOT any international English variety — this is monolingual US General American throughout
· ZERO rising sentence endings — every declarative statement closes with a falling pitch; upspeak signals uncertainty; this voice never seeks approval
· ZERO glottal stops replacing /t/ or /k/ — every final consonant is fully articulated
· ZERO h-dropping — "have" never sounds like "ave"
· ZERO vowel distortion of any kind — pure GA vowel inventory throughout

STUDIO AUDIO — ABSOLUTE PROHIBITIONS:
· Zero room reverb of any kind — no ambient acoustic environment, no decay, no tail
· Zero echo, flutter echo, or reflection artifacts
· Zero background noise — not traffic, not room tone, not HVAC, not any environmental sound
· Zero audio processing artifacts — no compression pumping, no noise gate clipping, no EQ ringing
· Zero proximity variation — the voice must maintain consistent acoustic character throughout; no fade on sentence endings

CHARACTER IDENTITY — ABSOLUTE LOCK — ZERO DRIFT PERMITTED:
· ZERO alteration to the target character's face geometry, bone structure, skin tone, hair, or any distinguishing feature across ANY frame of this scene
· ZERO face substitution — the face visible in frame 1 and the face visible in the final frame are identical in bone structure; they are the same human being at the same age with the same features
· ZERO AI beautification, skin smoothing, or idealization — real skin with visible pores, natural subsurface scattering, natural skin texture; NOT a perfected synthetic surface
· ZERO CGI sheen — no plastic luminosity, no synthetic specularity, no rendered-skin quality of any kind

⚠ ZERO WARPING — ZERO MORPHING — ZERO FACE DISTORTION — THIS IS ABSOLUTE:
The facial geometry is a fixed physical structure that does not change shape during speech or across frames. The jaw moves anatomically (up/down within natural range, driven by the masseter and digastric muscles). The lips move articulatorily (contact, separation, and reshaping consistent with the phonemes being produced). The face muscles produce expressions within the normal range of this specific human face. That is all. What is explicitly FORBIDDEN:
· ZERO jaw warping — the mandible does not stretch, elongate, compress, or distort beyond anatomical jaw movement; if the jaw appears to warp or deform rather than articulate, the scene has failed
· ZERO facial geometry drift — the distance between the eyes, the width of the nose, the position of the mouth relative to the nose and jaw — all fixed across all frames; no drift, no frame-to-frame geometry variation
· ZERO skin stretching artifacts — the skin over the jaw and cheeks stretches naturally over moving bone; it does NOT produce visual warping, smearing, or texture discontinuities
· ZERO temporal face morphing — the face at second 2.0 has identical bone structure to the face at second 0.5 and second 7.5; no slow drift of any feature across the scene duration
· ZERO face-swap artifacts — no frame where the face appears composited, pasted, or placed rather than continuous with the body; the head and body are one physically continuous unit
· ZERO uncanny valley distortion — if any frame produces a face that triggers discomfort, wrongness, or unease in a viewer, it contains a distortion artifact; eliminate it
Natural = anatomical jaw movement during speech (up/down), natural facial muscle expression within this character's normal range, natural skin deformation over moving bone
Unacceptable = any face geometry change beyond what human anatomy produces during speech

IMAGE REFERENCE AUTHORITY — REINFORCED:
· The TARGET CHARACTER images are the absolute identity authority; any deviation from the face in those images in any frame constitutes failure
· No substitute background — environment matches character photos exactly; zero environmental deviation
· Image 1 and Image 2 are the opening and closing frame blueprints; they are replicated, not interpreted

CAMERA — WHAT BREAKS THIS SCENE:
· Any camera movement that announces itself — visible zooms, slider passes, handheld instability, orbital moves
· Rack focuses during speech — the focal plane is set at scene start and does not move
· Movement faster than the pace of a listener leaning slightly forward — if it can be identified as camera motion, it is too much
· Any framing choice that competes with the subject for visual attention

PERFORMANCE — WHAT BREAKS THIS SCENE:
· Performed confidence instead of actual confidence — if it looks like acting, the scene has failed
· Nervous energy: fidgeting, rapid blinking (more than 8 per minute), unmotivated weight-shifting
· Theatrical emotion — the face announcing the feeling rather than containing it
· Over-choreographed gestures arriving before the thought — gesture must follow impulse, never precede it
· Monotone delivery — vocal variety is the engine of engagement; a flat voice destroys the scene
· Synchronized blinks — natural blinks have micro-variation in timing; identical bilateral blinks signal CGI
· Suppressed breath — the chest must show natural, visible breathing throughout
· Robotic stillness between gestures — stillness must have the ease of composure, not the rigidity of a held position

══════════════════════════════════════════════════
GENERATION DIRECTIVE:
══════════════════════════════════════════════════

Write the six-section VEO 3.1 prompt now. As you write, hold these priorities in strict hierarchy:

PRIORITY 1 — IDENTITY LOCK: The person rendered must be indistinguishable from the TARGET CHARACTER images. Face geometry, skin physics, bone structure, hair, wardrobe, environment — locked. If identity drifts by a single feature, the prompt has failed regardless of everything else.

PRIORITY 2 — LIP SYNC FIDELITY: The mouth must produce the specific articulatory geometry of each word in this script. Pre-speech behavior rendered. Post-speech settle rendered. Every bilabial closed. Every stressed vowel at full jaw width. Synchronization between audio and lip movement: zero perceptible error. Write the Lip Architecture section as if it is the only section that matters.

PRIORITY 3 — PERFORMANCE AUTHENTICITY: The face, body, and voice must read as a real human being in a real moment. Thought arrives before word. Breath is visible. Stillness has weight. Emotion is contained, not performed. The UHNWI viewer's nervous system must register "genuine" before their conscious mind has time to evaluate.

PRIORITY 4 — ACCENT AND AUDIO: US General American. Fully rhotic. Falling pitch. Zero music. Zero ambient sound. Studio-dry acoustic environment. If the accent slips for a single phoneme, the video fails.

PRIORITY 5 — CINEMATIC INTEGRATION: Camera, lighting, framing, and temporal continuity serve the performance — never compete with it. The visual world is locked from established constants. The camera earns every millimeter of movement.

For each section: open with the single most important rendering instruction. Close with the specific test of failure — what, if absent, would break the section's contribution to photorealism. Between opening and close: flowing, physical, director-register prose. No bullet points within sections. No headers within sections. One continuous voice per section, as if whispered to a cinematographer who already knows the craft and needs only the specific vision for THIS scene.

The standard: an affluent real estate investor watches this scene, pauses, and watches it again — not because of production quality, but because what they witnessed felt so genuinely human that they forgot they were watching a generated video. They felt addressed by a peer. They received something real. They want to hear what comes next.

Write to that standard. Six sections. Now.
`;

  const parts: any[] = [
    { text: prompt },
    { inlineData: { mimeType: inframeImage.type  || 'image/jpeg', data: inframeB64  } },
    { inlineData: { mimeType: outframeImage.type || 'image/jpeg', data: outframeB64 } },
    ...charBase64s.map((b64, i) => ({
      inlineData: { data: b64, mimeType: targetCharacterImages[i].type || 'image/jpeg' }
    }))
  ];

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts }],
    config: {
      systemInstruction: VEO_ENGINEER_SYSTEM_INSTRUCTION,
      thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH }
    }
  });

  const veoText = (response as any).text ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '';
  if (!veoText.trim()) throw new Error('engineerScenePrompt: model returned empty response. Please retry.');
  return veoText;
};

// ============================================================
// FUNCTION 5 — Refine Scene Prompt (Feedback Loop)
// Takes an existing prompt + user feedback → surgically upgraded prompt.
// ============================================================
export const refineScenePrompt = async (
  originalPrompt: string,
  scene:          ScriptScene,
  feedback:       string
): Promise<string> => {

  const annotatedScript = buildAnnotatedScript(
    scene.script_text,
    scene.acting_blueprint.emphasis_words || [],
    scene.acting_blueprint.pause_map      || []
  );

  const prompt = `
You are the world's finest VEO 3.1 prompt engineer, operating in REFINEMENT MODE. You have generated a prompt for a ${scene.role} scene, and the user has seen the VEO output and has specific feedback. Your task: surgically upgrade the prompt to address every issue — improving what failed while preserving everything that worked.

One refinement cycle typically closes 80% of the gap between a good first take and an excellent final take. This is that cycle.

══════════════════════════════════════════════════
ORIGINAL PROMPT:
══════════════════════════════════════════════════
${originalPrompt}

══════════════════════════════════════════════════
USER FEEDBACK — what to fix in the VEO output:
══════════════════════════════════════════════════
${feedback}

══════════════════════════════════════════════════
SCENE CONTEXT:
══════════════════════════════════════════════════
Scene #${scene.scene_number} — "${scene.title}"
Role: ${scene.role} | ${scene.duration_seconds}s | Energy: ${scene.energy_level}/10
Script: "${scene.script_text}"
Annotated script: "${annotatedScript}"

══════════════════════════════════════════════════
SURGICAL REFINEMENT PROTOCOL:
══════════════════════════════════════════════════
1. Diagnose which section(s) caused the failure:
   — Character drift / wrong face / wrong environment → fix Character section with stronger identity anchoring
   — Lip sync issues on a specific word → fix Lip Architecture section; name that exact word and its phoneme geometry explicitly
   — Too much camera movement → fix Shot section camera doctrine to explicitly prohibit the movement seen
   — Wrong emotion / face too expressive → fix Performance section containment level
   — Wrong accent / rising inflection → fix Voice section with specific counter-directive
   — Subtitles appeared → add explicit ironclad prohibition in Script section and Do Not Include
   — Wrong energy (too fast / too slow) → fix Voice pacing direction and Performance energy level
   — CGI skin / artificial look → fix Character section with stronger SSS and pore texture directives
   — Wrong gesture timing → fix Performance section gesture-follows-thought directive

2. Rewrite the implicated section(s) with the specific fix — more precise, more directive, more concrete

3. Preserve every section that was not implicated — do not weaken what was working

4. If the feedback mentions a specific word (e.g. "lip sync on 'capital'"), add a dedicated paragraph in Lip Architecture naming that word, its dominant phoneme, and the exact mouth geometry required

5. For every fix: explain to VEO WHY this matters — not just what to do but the felt consequence if it fails

══════════════════════════════════════════════════
ABSOLUTE LAWS — remain in every version, no exceptions:
══════════════════════════════════════════════════
ZERO MUSIC. ZERO AUDIO EFFECTS. ZERO AMBIENT SOUND.
ZERO SUBTITLES. ZERO CAPTIONS. ZERO ON-SCREEN TEXT OF ANY KIND.
Script is SPOKEN ONLY — never displayed visually anywhere in the frame.

Output the complete refined 6-section VEO prompt. All six sections must be present: Character, Shot, Performance, Lip Architecture, Voice, Script.
No preamble. No explanation of your changes. Just the refined prompt.
`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  const refined = (response as any).text ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text ?? '';
  return refined.trim() || originalPrompt;
};

// ============================================================
// FUNCTION 6 — Re-Annotate Script (Prosodic Precision Pass)
// Dedicated Gemini call to place emphasis + pause markers with
// phonemic precision, informed by gravity center and vocal gear.
// ============================================================
export const reAnnotateScript = async (
  scene:             ScriptScene,
  referenceAnalysis: ReferenceAnalysis
): Promise<{ emphasis_words: string[]; pause_map: string[] }> => {
  const gearLabel = (scene.acting_blueprint as any).vocal_gear === 4 ? 'GEAR 4 — Passionate/Driving'
    : (scene.acting_blueprint as any).vocal_gear === 3 ? 'GEAR 3 — Engaged/Clear'
    : (scene.acting_blueprint as any).vocal_gear === 2 ? 'GEAR 2 — Conversational/Warm'
    : 'GEAR 1 — Intimate/Quiet';

  const gravityCenterWord = (scene as any).gravity_center_word
    || scene.acting_blueprint.emphasis_words?.[0]
    || 'the key phrase';

  const prompt = `You are an elite prosodic director. Return the most natural, impactful prosodic annotation for this scene — the annotation that produces GENUINE authority delivery, not performed delivery.

SCENE: "${scene.title}" | ${scene.role} | ${gearLabel} | ${scene.duration_seconds}s | Energy ${scene.energy_level}/10
Gravity Center: "${gravityCenterWord}" | Tone: ${scene.emotional_tone}
SCRIPT: "${scene.script_text}"

EXISTING (refine, don't copy): emphasis=${JSON.stringify(scene.acting_blueprint.emphasis_words || [])} pauses=${JSON.stringify(scene.acting_blueprint.pause_map || [])}

RULES FOR NATURAL DELIVERY:
1. EMPHASIS WORDS — 2 to 4 only. Fewer is more powerful. More than 4 makes everything equally stressed, which means nothing is emphasized.
   — "${gravityCenterWord}" MUST be first (this is the word the whole scene builds toward)
   — Then: words with bilabial sounds (p/b/m) for lip sync visibility, OR wide-vowel words (jaw drop visible), OR the single word that carries the scene's emotional weight
   — NEVER mark filler words, conjunctions, or transitions

2. PAUSE MAP — 1 to 3 pauses only. Real authority does not pause constantly.
   — MANDATORY: one pause of 1.0–2.0s after "${gravityCenterWord}" — this is the silence that lets the gravity center land
   — OPTIONAL: one short pause (0.3–0.5s) at a natural sentence boundary if the script has two distinct ideas
   — NEVER: pauses after every clause, pauses before every emphasis word, mechanical gaps
   — Format: "Xs after 'word'" or "Xs before 'phrase'" (numeric seconds, exact word from script)

Return ONLY valid JSON:
{
  "emphasis_words": ["gravityWord", "word2"],
  "pause_map": ["1.5s after 'gravityWord'", "0.4s after 'sentence-end-word'"]
}`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  const raw = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '';

  try {
    const result = safeJsonParse<{ emphasis_words: string[]; pause_map: string[] }>(raw, 'reAnnotateScript');
    return {
      emphasis_words: Array.isArray(result.emphasis_words) ? result.emphasis_words.slice(0, 6) : [],
      pause_map:      Array.isArray(result.pause_map)      ? result.pause_map.slice(0, 5)      : [],
    };
  } catch {
    return {
      emphasis_words: scene.acting_blueprint.emphasis_words || [],
      pause_map:      scene.acting_blueprint.pause_map      || [],
    };
  }
};

// ============================================================
// FUNCTION 6b — Build Character Physical Lock
//
// Forensic prose extraction from the 3 uploaded character photos.
// Called ONCE per session before the first scene is engineered.
// The output is cached in AppState and injected into EVERY scene's
// BINDING CONSTRAINTS as a locked physical identity specification.
//
// WHY separate from extractCharacterIntelligence (which outputs JSON
// for the image-gen pipeline): VEO needs PROSE, not JSON. VEO renders
// from BOTH the attached image AND the text description simultaneously.
// If the text description is generic ("elite advisor") while the photo
// shows a specific person, VEO averages toward a generic face. This
// function produces forensic prose that anchors VEO to the EXACT person
// in the photos.
// ============================================================
export const buildCharacterPhysicalLock = async (
  targetCharacterImages: File[]
): Promise<string> => {
  if (targetCharacterImages.length === 0) return '';

  const ai = getAI();
  const charBase64s = await Promise.all(targetCharacterImages.map(fileToBase64));

  const parts: any[] = [];
  for (let i = 0; i < charBase64s.length; i++) {
    const raw = charBase64s[i].includes(',') ? charBase64s[i].split(',')[1] : charBase64s[i];
    parts.push({ inlineData: { data: raw, mimeType: targetCharacterImages[i].type || 'image/jpeg' } });
  }

  const n = targetCharacterImages.length;
  parts.push({ text: `You are a forensic identity specialist, cinematographer, and VEO 3.1 expert. ${n} photo${n > 1 ? 's of the SAME person are' : ' is'} attached. This person — exactly as they appear, in exactly the environment visible in these photos — must appear identically in every frame of an AI-generated video.

Your task: produce a CHARACTER PHYSICAL IDENTITY LOCK in TWO PARTS.

PART 1 — VISUAL ANCHOR CHECKLIST (structured, numbered, quote-ready):
This section will be injected verbatim into BINDING CONSTRAINTS. It must be a numbered checklist of every non-negotiable visual element. VEO will be instructed to verify ALL items before rendering. Be maximally specific. Every claim must be directly observable in the attached photos. No inferred details. No generic descriptions.

Format EXACTLY as follows (keep the bracket numbers, colons, and category labels):

VISUAL ANCHOR CHECKLIST — VEO MUST VERIFY ALL 8 BEFORE RENDERING:
[1] FACE/JAW: [Describe the jaw terminus precisely — sharp-angled / squared / soft rounded — the jawline definition from ear to chin, cheekbone height and projection angle, chin shape and fleshy or defined quality, philtrum width, forehead geometry]
[2] EYES: [Exact iris color in optical terms — not "brown" but "warm amber-brown with dense chestnut outer ring and lighter honey interior zone." Iris fibrous structure. Limbal ring prominence. Eye shape — almond, hooded, deep-set, openly rounded. Any asymmetry between left and right eye]
[3] HAIR: [Exact hair color as optical description under the photo lighting. Texture — fine/coarse, straight/wavy. Density and hairline character. Where specular highlights fall. Whether individual strands are visible at perimeter or it reads as a unified mass]
[4] SKIN TONE: [Exact undertone in optical terms — "warm amber-olive with visible subsurface scatter at nasal tip and ear cartilage," "cool porcelain with faint blue cast in shadow zones." T-zone vs. lateral cheek sheen differential. Any visible pore texture or vellus hair bloom]
[5] WARDROBE: [List every visible garment exactly — type (shirt/jacket/collar), specific color under the photo lighting, visible fabric character (matte/sheen, weight), any visible buttons, collar style, or pattern. This must be specific enough that a costume department could recreate it exactly]
[6] BACKGROUND/ENVIRONMENT: [Describe EXACTLY what is visible behind the presenter in the photos — specific colors, gradients, any visible architectural elements, depth, whether it is a studio backdrop, room interior, outdoor space, or neutral gradient. Include the specific hue and tone of the background]
[7] LIGHTING: [Describe the key light direction (from which side, at what height), its color temperature (warm/neutral/cool), the shadow character it creates on the face, whether there is a visible rim/hair light, and the overall ambient light quality in the photo environment]
[8] MOST DISTINCTIVE FEATURE — IDENTITY TEST: [Name the single most identifying physical feature of this person — the one characteristic whose absence would immediately signal that VEO rendered a different person. Be maximally specific: not "strong jaw" but "a characteristically squared jaw terminus with a specific flatness at the chin base that creates a rectangular lower face geometry"]

PART 2 — FORENSIC IDENTITY PROSE (6 paragraphs, for [SUBJECT & ORGANIC PHOTOREALISM] section):
Cross-reference ALL ${n} photo${n > 1 ? 's' : ''} simultaneously — areas where photos differ reveal the true face geometry more accurately than any single image. Dense, flowing prose. No headers. No bullet points. Only observably true claims from the attached photos.

PARAGRAPH 1 — FACE GEOMETRY & IDENTITY ANCHORS: The jaw's exact terminus, how strongly the jawline defines from ear to chin, whether it is prominent or gentle. The cheekbones — height on the face, projection from face plane, the specific angle at which they catch key light. Orbital ridge and brow prominence — whether they create visible shadow over the eyes. Nasolabial depth at rest. Philtrum width. Chin projection and fleshy/defined quality. Name the 5 most distinctive features — the physical characteristics that most distinguish this person and MUST be reproduced in every frame.

PARAGRAPH 2 — SKIN PHYSICS: Exact skin tone in optical/undertone terms. Subsurface scattering quality — warm glow color at nasal tip, ear cartilage, nasolabial folds, cheekbone edges. T-zone specular vs. lateral cheek diffuse differential. Fresnel brightening at jaw edge, temples, orbital rim. Any vellus hair bloom. Pore texture under key light. Skin deformation quality on jaw movement — fleshy and soft, or taut and structured.

PARAGRAPH 3 — EYES, BROW & GAZE: Exact iris color in precise optical terms. Iris fibrous structure density. Limbal ring prominence. Eye shape. Brow architecture — distance from eye, curvature, natural arch height, density and color. Shadow depth in orbital cavity. The precise quality of this person's gaze at biological rest — what the eyes communicate before any expression is applied.

PARAGRAPH 4 — HAIR: Exact hair color as optical description. Texture and weight. Density and hairline character. Where the key light creates specular highlights vs. absorption. Individual strand resolution at the perimeter.

PARAGRAPH 5 — BUILD, WARDROBE & PRESENCE: Visible build — shoulder width and set, neck length, how this person holds their weight and posture. Exact garment description — type, fabric quality, color under photo lighting, collar/lapel character. What the wardrobe signals. Any distinguishing marks, asymmetries, or characteristic features not yet mentioned.

PARAGRAPH 6 — FORENSIC IDENTITY STATEMENT + ENVIRONMENT: Three sentences establishing the definitive forensic identity: (1) the face's most immediately readable quality in the first 0.3 seconds; (2) the physical characteristics hardest to approximate — the specific geometry that makes this person uniquely themselves; (3) what would be wrong if VEO rendered a slightly different person — the specific features whose absence signals identity drift. Then: two additional sentences describing the ENVIRONMENT — the exact background color, depth quality, and lighting character visible in the photos, and why reproducing it exactly (not approximating it) is essential to scene consistency.

Write PART 1 (checklist) followed immediately by PART 2 (prose). Start PART 1 with "VISUAL ANCHOR CHECKLIST" on the first line. Start PART 2 with "FORENSIC IDENTITY PROSE:" on its own line. No other preamble.` });

  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } },
  });

  const text = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '';
  return text.trim();
};

// ============================================================
// FUNCTION 7 — Compute Phonemics (Lip-Sync Pre-Computation)
// Dedicated Gemini call for word-by-word phonemic analysis.
// Output injected directly into Lip Architecture section.
// ============================================================
export const computePhonemics = async (
  scene: ScriptScene
): Promise<string> => {
  const prompt = `You are a master VEO 3.1 lip-sync director. Your task: produce rendering-ready prose guidance that tells VEO exactly what the mouth does on the most visually critical words in this script. Write in the same director-register prose that VEO understands — physical, specific, felt from inside the performance. Never use IPA slash notation or phonetic alphabet symbols. Write as a cinematographer directing a close-up: "the lips press completely together on 'market', then spring apart as the jaw drops 8mm into the open vowel."

SCRIPT: "${scene.script_text}"
Role: ${scene.role} | ${scene.duration_seconds}s | ${scene.word_count} words

Write 4 short prose paragraphs — no headers, no bullet points, no lists:

PARAGRAPH 1 — THE HIGHEST-RISK WORD: Identify the single word in this script most likely to fail lip sync (dense consonant cluster, a word starting mid-syllable, or a sibilant-heavy word). Describe exactly what the mouth does: the precise sequence of lip, jaw, and tongue positions, the jaw travel in millimeters, the precise moment of closure and release, the transition into the vowel. Make it specific enough that VEO cannot approximate.

PARAGRAPH 2 — BILABIAL MOMENTS: For every word in this script that requires full lip closure (words containing the sounds written as "p", "b", or "m" in English), describe the full contact-to-release sequence: the lips pressing completely together with zero visible gap, the brief held closure (longer on stressed syllables), then the release — the jaw falling open and lips parting simultaneously into the following vowel. Name each word. Make the zero-gap requirement unmistakable.

PARAGRAPH 3 — CALM DELIBERATE FLOW: Describe how this specific sentence sounds when spoken by someone in a state of complete calm authority — no urgency, no push, jaw returning to gravitational rest between every stressed word. At deliberate pace, describe: where the jaw stays at its floating rest position (function words, conjunctions, prepositions — the jaw barely opens), versus where it opens fully (stressed vowels on key words — the jaw drops as the voice arrives, not reaches). Describe the rhythm of jaw movement as a breathing, floating pattern — not word by word but as a continuous gravitational cycle of release and return. Identify one specific word boundary where the jaw must carry forward without resetting fully to neutral — the co-articulation point where one word's closing mouth shape becomes the next word's opening shape. Describe this handoff exactly.

PARAGRAPH 4 — THE OPENING AND CLOSING: Describe exactly what the mouth does in the 0.3s before the first word begins (jaw position, lip state, whether the mouth is parted or closed) and what happens after the last word completes (jaw relaxing, lips easing toward rest over 0.4s, the specific closed or open state the mouth settles into). These two moments are the most reliable lip-sync failure points in AI generation.

Write all four paragraphs now. No preamble. No section headers. Prose only. Each paragraph 2-4 sentences.`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  const text = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '';
  return text.trim();
};

// ============================================================
// FUNCTION 9 — Distill VEO Prompt (Final Prompt Optimiser)
//
// Takes the full 5,000+ word engineered prompt (Pass 2) and
// distils it to a ≤800-word precision concentrate.
//
// RATIONALE: VEO 3.1 processes prompts sequentially and weights
// the first and last sentences of each section most heavily.
// A 5,000-word prompt dilutes this signal — VEO's attention is
// spread across elaboration, explanation, and repetition that
// adds no rendering value. A 700-800 word distillation in which
// EVERY sentence is a direct rendering instruction concentrates
// the signal: more physics, less prose, better output.
//
// What survives distillation:
//   • All identity + audio binding constraints (compressed)
//   • The highest-physics Character sentences (SSS, Fresnel,
//     catchlights, jaw deformation) — scene-specific only
//   • Shot: opening frame match, camera doctrine, closing frame
//   • Performance: moment-before, charisma quality, through-action,
//     lean-in signal, the director's single best line
//   • Lip Architecture: pre-speech onset + the 2-3 most critical
//     articulatory moments for THIS script specifically
//   • Voice: resonance gear, pace spec, accent lock, emphasis arch
//   • Script: annotated script verbatim + delivery map (unchanged)
//
// What gets cut: repetition, elaboration, explanation of why,
// generic directives that apply to any scene rather than this one.
// ============================================================
export const distillVeoPrompt = async (
  fullPrompt:            string,
  scene:                 ScriptScene,
  targetCharacterImages?: File[]
): Promise<string> => {

  // Convert up to 3 target character photos to base64 for identity verification
  const charBase64s: string[] = targetCharacterImages && targetCharacterImages.length > 0
    ? await Promise.all(targetCharacterImages.slice(0, 3).map(img => fileToBase64(img)))
    : [];

  const hasCharPhotos = charBase64s.length > 0;

  const prompt = `You are the world's most precise VEO 3.1 prompt distiller. You have received a fully engineered VEO 3.1 prompt.${hasCharPhotos ? ` ${charBase64s.length} TARGET CHARACTER photo(s) are attached to this message — these photos define the EXACT physical appearance of the person who must appear in the video. Every physical description in your output must match the person in these photos. If there is any conflict between the source prompt and the photos, the PHOTOS WIN.` : ''}

Your task: produce a ≤1,200-word structured distillation by filling 6 sections with specific named content from the original. This is density-targeted extraction, not word-count compression — the goal is maximum rendering signal per word, not minimum word count. Keep every sentence that gives VEO a specific physical mechanism to render. Eliminate only: (a) restatements of the same directive, (b) "why" explanations rather than "what to render," (c) directives that apply to any scene rather than this specific one.

FULL ENGINEERED PROMPT (source material — extract from this):
════════════════════════════════════════════════════════════
${fullPrompt}
════════════════════════════════════════════════════════════

SCENE CONTEXT:
Script: "${scene.script_text}"
Role: ${scene.role} | ${scene.duration_seconds}s | ${scene.word_count} words
Gravity center word: "${(scene as any).gravity_center_word || scene.acting_blueprint.emphasis_words?.[0] || 'key insight'}"
Emphasis words: ${JSON.stringify(scene.acting_blueprint.emphasis_words || [])}
Pause map: ${JSON.stringify(scene.acting_blueprint.pause_map || [])}

════════════════════════════════════════════════════════════
DISTILLATION INSTRUCTIONS — fill each slot exactly:
════════════════════════════════════════════════════════════

OUTPUT FORMAT — produce exactly this structure (section headers verbatim, content from slots):

LOCK: [SLOT L — one sentence: character token + "US General American, zero music/effects/subtitles/ambient sound, Image 1 opens exactly, Image 2 closes exactly${hasCharPhotos ? ', TARGET CHARACTER photos attached define every physical attribute' : ''}." Use the exact character token from the source prompt.]

Character:
[SLOT C1 — Identity sentence: ${hasCharPhotos ? 'Describe the person in the attached TARGET CHARACTER photos — their exact face structure, skin tone, eye color, hair, and defining physical features — in one precise sentence. These photos ARE the character; override any conflicting description in the source.' : 'who this person is physically, in one precise sentence that makes them unmistakable. From the Character section of the source.'}]
[SLOT C2 — The single best skin physics sentence from the source: the one involving subsurface scattering, Fresnel reflectance, sebum differential, or vellus hair — whichever is most specific and renders most directly. One sentence only.]
[SLOT C3 — Living eyes: primary warm key catchlight (upper iris, warm) + secondary cool fill catchlight (opposite side, smaller) + limbal ring (dark graduated band at iris edge, depth not a hard line) + this mandatory phrase: "Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts) — eyes continuously alive, never fixed." One-two sentences.]
[SLOT C4 — Jaw deformation sentence: how the skin and muscle deform during speech articulation for this specific character. One sentence.]
[SLOT C5 — Wardrobe sentence: fabric weight, specific color under this light temperature, how it drapes. One sentence.]
[SLOT C6 — FACIAL GEOMETRY LOCKS — the four measurements VEO needs to hold identity consistency across scenes. Extract these from the source prompt's facial geometry description, or derive them from the TARGET CHARACTER photos if present. Write as a single prose sentence containing all four: (1) interocular distance character ("wide-set / average / close-set — X mm equivalent at this focal length"), (2) nasolabial fold depth at neutral expression ("shallow / moderate / deep trace"), (3) philtrum proportion + cupid's bow ("long/short philtrum, defined/soft cupid's bow"), (4) mandible angle ("soft-round / square-defined / tapered"). Example: "Wide-set eyes — 36mm equivalent at 85mm focal length; shallow nasolabial traces at neutral; medium-length philtrum with a soft cupid's bow; mandible angle square-defined with a clean jawline terminus." One sentence. Every scene must reproduce these four measurements identically — they are the identity lock VEO drifts on most.]

Shot:
[SLOT S1 — Opening frame: "The video opens matching Image 1 exactly." + the inframe prose description from the source (body position, expression, mouth state). Two sentences max.]
[SLOT S2 — Camera doctrine: the camera movement instruction for this specific scene role + lens spec + focal length psychology for this scene's emotional register. Two sentences max.]
[SLOT S3 — PHYSICS DIFFERENTIAL — the delta between Image 1 and Image 2. Do NOT describe "what happens." Describe the exact state changes: which specific things change from opening to closing frame (head angle shift in degrees, lean direction and amount, expression quality transition, jaw position), AND which things remain constant (background, framing, lighting character). Format: "CHANGES: [list]. CONSTANTS: [list]." This is a closed loop — VEO reads Image 1, applies the delta, arrives at Image 2. Two sentences max.]
[SLOT S4 — Closing frame: "The video ends matching Image 2 exactly." + the outframe prose description. Two sentences max.]

Performance:
[SLOT P1 — Moment-before (⚡ EXEMPT FROM WORD COUNT — preserve in full, do not compress): copy the EXACT moment-before text from the source verbatim, or if the source has only brief moment-before language, expand it to full specificity using the character DNA and scene context. This slot MUST contain 6-8 sentences covering: (1) jaw position and lip gap 0.5s before first word, (2) breath state — chest position, inhale or exhale, (3) eye quality and gaze character, (4) psychological texture — what thought is happening internally, (5) weight distribution through spine and shoulders, (6) the specific micro-tension or micro-release in the face that signals this particular pre-speech state, (7) what the face reveals about what is about to be said without saying it, (8) the transition moment — when preparation ends and the first phoneme begins. The moment-before is the single most important slot for photorealistic performance. Do not compress. If the source has 6-8 sentences of pure physical specificity here, copy them character-for-character.]
[SLOT P2 — Through-action: the active verb phrase from the source, including its master arc reference if present. One sentence.]
[SLOT P3 — Charisma quality: the specific magnetic quality of this scene — what makes it irresistible. From the CHARISMA QUALITY or CHARISMA ARCHITECTURE section. One sentence.]
[SLOT P4 — Subtext layer: what the face says beneath the words. From the SUBTEXT LAYER section. One sentence.]
[SLOT P5 — Role direction: the single best directive sentence from this role's performance direction. The one a director would say in the final whisper before the take. One sentence.]
[SLOT P6 — The director's whisper: the DIRECTOR'S WHISPER sentence verbatim from the source.]

[SLOT P7 — ⚡ EXEMPT FROM WORD COUNT — RELAXED BIOLOGICAL FOUNDATION for this scene. This is the non-negotiable perceptual bedrock. Every other performance directive is built on top of this. Write in biological/experiential language ONLY — no Latin muscle names, no mm/degrees. Calibrate to ${scene.emotional_tone} at energy ${scene.energy_level}/10.

GOVERNING STATEMENT: One direct sentence establishing that this face is at genuine biological ease in every frame — nothing performed, nothing held, everything organically present.

BIOLOGICAL BASELINE (describe each using experiential, sensory language):
· BROW: [what does the brow look like — smooth and unhurried, or is there any specific quality? For ${scene.emotional_tone}: describe the exact surface quality of the forehead. No muscle names.]
· INNER BROW SPACE: [the territory between the brows — wide open and untroubled, or specifically what quality? Describe how ${scene.emotional_tone} specifically manifests here — what does certainty look like in this space vs. what anxiety would look like]
· CHIN: [the fleshy quality of the chin and lower face — completely still and soft, or what? For ${scene.emotional_tone}: describe the specific quality of biological ease here; why its complete softness reads as deep security]
· JAW: [soft and floating, or specific quality? Describe the fleshy organic weight of a relaxed jaw for this scene's emotion; the natural slight asymmetry of jaw-at-rest; calibrate to ${scene.energy_level}/10 energy]
· LIPS: [at soft organic rest between words — the natural asymmetry, the specific quality of ease in this emotional register; not compressed, not controlling]
· EYES: Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts). [Plus: the specific quality of eye aperture and engagement for ${scene.emotional_tone} at ${scene.energy_level}/10 — what the eyes carry that makes this specific emotional state legible without announcement]
· CHEEK WARMTH: [Duchenne (eye corners + cheeks together, involuntary) if genuine warmth is present in ${scene.emotional_tone}; neutral-warm (cheeks at soft organic rest) if authority-dominant — specify which and describe what it looks like biologically]

FREEZE-FRAME TEST: any random frame examined by a sophisticated viewer must register as "genuinely confident, relaxed, present." State: PASS or the specific biological adjustment needed.]

[SLOT P8 — ⚡ EXEMPT FROM WORD COUNT — CHARISMA ARCHITECTURE (built on the relaxed foundation above): For each layer, write ONE sentence giving VEO the exact physical mechanism — not what the layer IS, but what it looks like in this specific face at this specific moment. Charisma is only magnetic when built on a relaxed body; if the foundation (P7) is tense, these layers become performed charisma, which repels.
LAYER 1 — IRRESISTIBLE THOUGHT: [what the eyes carry that makes the viewer feel the character knows more — specific orbital and gaze quality that signals depth without announcing it]
LAYER 2 — INHABITED PAUSE: [the micro-sequence of pause behavior: the release of the completed phrase (eyes ease), the reception of the next thought (micro-inward shift 0.1s), the fractional brightening before the next word — specific muscle and eye quality]
LAYER 3 — SELECTIVE DISCLOSURE: [the specific quality in the eyes of someone selecting carefully from a larger reserve — precise orbital state; the contained fullness that reads as authority]
LAYER 4 — EARNED VULNERABILITY: [the precise moment and micro-release — which muscle fractionally softens, for how long, around which specific word in this script "${scene.script_text.trim().split(/\s+/).slice(0, 4).join(' ')}..."]
LAYER 5 — LEAN-IN SIGNAL: [the exact forward directionality in the frame — not a physical lean, but the quality of directed energy toward the lens; the specific combination of body, eye, and pace that creates this signal]
SUBTEXT LAYER: [what the face says beneath the words — the specific physical expression of: "${(scene as any).subtext || (scene.acting_blueprint as any).subtext || 'the authority of someone who has verified what they are saying in real rooms with real stakes'}"]
BETWEEN-PHRASE FACE: [what the face does in the 0.3-0.8 seconds between phrases — the echo of what was just said + the arrival of what comes next; described as specific orbital, jaw, and brow micro-behavior — this is where charisma actually lives]]

[SLOT P9 — ANTI-ACTING DIRECTIVES (3 most critical for this scene's role): From the natural performance architecture, identify the 3 behavioral signals most important for this specific ${scene.role} scene to feel authentic rather than AI-generated. For each: name the signal and give the specific physical mechanism VEO must render.
1. THOUGHT-BEFORE-WORD: [specific description — the 0.05-0.15s window where the thought has arrived in the face before the voice engages; what specific micro-movement precedes the first phoneme of "${scene.script_text.trim().split(/\s+/).slice(0, 3).join(' ')}..."]
2. INTER-WORD FACE: [what the face does in the 50-100ms between the key words in this script — the specific micro-expression the face makes while the next thought arrives]
3. [Third signal most relevant to ${scene.role}: choose from MICRO-ASYNCHRONY (dominant/non-dominant face leads fractionally), BREATH-DRIVEN POSTURAL LIFE (sternum and shoulder micro-movement with each breath), UNPLANNED MICRO-CORRECTIONS (fractional head adjustment mid-phrase), or VARIABLE BLINK RHYTHM (irregular blink intervals 2-8 seconds) — whichever is most critical for this role]]

[SLOT P10 — PERFORMANCE STANDARD DECLARATION: Write exactly: "HOLLYWOOD HYPER-ULTRA-REALISM — this is the standard of Meryl Streep in Sophie's Choice, Daniel Day-Lewis in There Will Be Blood, Anthony Hopkins in The Silence of the Lambs. The face is the primary dramatic instrument. Every micro-millimeter of muscular activity is a performance decision. Every fraction of a second between thoughts is a dramatic beat. The camera happened to be present during a genuine internal transmission — nothing is performed because the camera is present, nothing is added for the lens." Then add one sentence: the FELT-DOCUMENTARY TEST for this specific scene — what a director would look for to confirm this scene passes (derived from the FELT-DOCUMENTARY STANDARD section of the source).]

Lip Architecture:
[SLOT LA1 — Speech onset FIRST — if speech_onset_phoneme is present in the source: copy it verbatim as the opening of this section, prefixed with "SPEECH ONSET — FIRST PHONEME OF [FIRST WORD]:" — this slot opens the section and VEO reads it first. If not present, start with the pre-speech behavior sentence instead.]
[SLOT LA2 — Highest-risk articulatory sequence: from the phonemic precomp or phonemic anchors — the word most likely to fail lip sync with its specific mouth geometry sequence. 2-3 sentences.]
[SLOT LA3 — Bilabial moments: every word in this script requiring full lip closure, with the contact-to-release sequence described. 1-2 sentences.]
[SLOT LA4 — Closing mouth state: how the mouth settles after the last word of this script. One sentence.]

Voice:
[SLOT V1 — Resonance gear: the RESONANCE ARCHITECTURE sentence for this scene's gear (GEAR 1/2/3/4). The full gear spec from the source — chest/mask ratio, Singer's Formant engagement. 2 sentences.]
[SLOT V2 — Pace + accent: the scene-specific pace calculation (WPM) + US General American rhotic spec. One sentence.]
[SLOT V3 — Emphasis delivery: the emphasis architecture for this script — how the emphasis words are delivered phonemically. One sentence.]
[SLOT V4 — Gravity center delivery: how the gravity center word is delivered — deceleration %, pitch, jaw, post-word silence. One sentence.]

Script:
[SLOT SC0 — ⚡ EXEMPT FROM WORD COUNT — TEMPORAL CHOREOGRAPHY: Break this ${scene.duration_seconds}s scene into chronological [Xs - Xs] action brackets. Calculate phrase timing from: 0.3s pre-speech inhale, then distribute speech duration (~${Math.round((scene.duration_seconds - 0.6) * 0.85 * 10) / 10}s for words, remainder for pauses), then 0.3s post-speech settle. Use the annotated script to identify phrase boundaries.

MANDATORY TEMPORAL BRACKETS:
[0.0s - 0.3s]: PRE-SPEECH INHALE — describe: chest rising with a quiet, organic breath; face inhabiting the moment-before state; lips at soft biological rest; body loaded with presence before any word arrives; natural moisture in the eyes; the specific organic stillness that is not emptiness but readiness

[Then for each phrase/sentence in the script: a bracket describing the organic facial behavior, body movement, and gestural life during that phrase — use biological language only]

[For each [PAUSE-Xs] marker: ACTIVE SILENCE — face is biologically alive: subtly swallowing, visibly processing the thought just given, a barely perceptible fleshy jaw micro-movement, eyes drifting briefly inward then returning as the next thought arrives; chest continuing its natural respiratory rhythm through the silence; the body never anatomically dead]

[Final bracket of ${scene.duration_seconds}s]: POST-SPEECH SETTLE — jaw returning to soft fleshy rest; lips easing toward natural, slightly asymmetric closure; face holding the emotional weight of what was just transmitted without resetting; the chest falling on a natural exhale; the period heard in the body's quiet return to organic stillness

BIOLOGICAL LANGUAGE RULES FOR ALL TEMPORAL BRACKETS:
· No anatomy names: "brow" not "frontalis", "jaw" not "masseter", "chin" not "mentalis"
· No measurements: "fleshy, natural jaw articulation" not "8mm jaw drop"
· No frozen language: never "locked", "unblinking", "rigid", "perfectly still"
· Eyes always: "natural moisture, spontaneous blinking, organic pupil dilation"
· Emphasis always: "voice naturally slowing/deepening" not "15-20% deceleration"
· Asymmetry always present: "naturally asymmetric," "slightly uneven," "organically off-center"]

[SLOT SC1 — Annotated script: copy the complete annotated script from the Script section of the source EXACTLY — every *emphasis* marker and every [PAUSE-Xs] marker preserved character-for-character. Do not alter one character.]
Accent: US GENERAL AMERICAN — every /r/ fully rhotic, falling pitch on every declarative, stress-timed rhythm. The natural educated American voice: authority without stiffness, ease without casualness.
Emphasis (*word*): the voice naturally slowing and deepening around this word — weight through completeness, not volume; the vowel opens to its full organic resonance; a brief natural silence follows as the word settles.
Pauses [PAUSE-Xs]: active biological silence at genuine cognitive boundaries — subtly swallowing, visibly processing, a barely perceptible fleshy jaw micro-movement, chest continuing its natural respiratory rhythm; face continuously alive, never anatomically dead.
Gravity center ("${(scene as any).gravity_center_word || scene.acting_blueprint.emphasis_words?.[0] || 'key word'}"): the voice naturally slowing the way all voices do around what they genuinely mean — lowest pitch, widest organic jaw opening, longest inhabited silence after.
Natural delivery: slightly faster on function words; naturally slower approaching the gravity center — the acoustic fingerprint of genuine authority; the voice of someone thinking about what they mean, not how to say it.
ZERO music, effects, subtitles, or ambient sound. Duration: ${scene.duration_seconds}s, ${scene.word_count} words.

════════════════════════════════════════════════════════════
SLOT-FILLING RULES:
════════════════════════════════════════════════════════════
1. Every slot must be filled — no slot may be empty or skipped.
2. Fill from the source prompt — do not invent content not present in the source.
3. If a slot asks for "one sentence," give exactly one sentence — not two.
4. If a slot asks to copy verbatim (Slot SC1, Slot P6, Slot LA1), copy character-for-character.
5. WORD COUNT: The Character + Shot + Voice + Script sections target ≤500 words total. The Performance section (P1-P10) has NO word limit — it is the soul of the scene. Slots P1, P7, P8, P9, P10 are EXEMPT from any word count pressure. Fill them completely. The Character and Shot sections may be tightened to create space; the Performance section must never be compressed.
6. COMPRESSION PRIORITY — if any section must be shortened to fit, cut in this order: (1) Voice section redundancies first, (2) Shot section elaborations second, (3) Character section secondary sentences third. NEVER compress the Performance section.
7. HARMONY CHECK — before completing, verify that all 6 sections describe the SAME person in the SAME moment: (a) Character's physical description matches the face implied by Performance's 7-muscle state; (b) Voice's resonance gear matches Performance's energy level; (c) Shot's camera distance matches the intimacy register in Voice; (d) Lip Architecture's pre-speech state matches Performance's moment-before. If any section contradicts another, harmonize it. A prompt where all sections agree amplifies itself; a prompt where sections contradict fragments the rendering.
8. No preamble, no word count declaration, no explanation. Start directly with LOCK:`;

  const ai = getAI();

  // Build parts array — text prompt first, then character photos (identity authority)
  const parts: any[] = [{ text: prompt }];
  for (const b64 of charBase64s) {
    // Strip data URL prefix if present (fileToBase64 returns full data URL)
    const rawB64 = b64.includes(',') ? b64.split(',')[1] : b64;
    parts.push({ inlineData: { data: rawB64, mimeType: 'image/jpeg' } });
  }

  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  const distilled = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '';
  return distilled.trim() || fullPrompt;
};

// ============================================================
// FUNCTION 8 — Critique VEO Prompt (Self-Improvement Pass)
// Identifies the 3 weakest sentences in a generated prompt
// and rewrites the complete prompt with those 3 elevated.
// This is the quality gate between a good first take and
// an excellent final take.
// ============================================================
export const critiqueVeoPrompt = async (
  generatedPrompt: string,
  scene:           ScriptScene
): Promise<string> => {
  const prompt = `You are the world's most demanding VEO 3.1 prompt quality auditor. A prompt has been generated for a ${scene.role} scene (${scene.duration_seconds}s, ${scene.word_count} words). Script: "${scene.script_text}"

Your task: run FOUR passes — (A) 5-sentence structural critique; (B) LIP SYNC AUDIT; (C) AUDIO FIDELITY AUDIT; (D) VISUAL FIDELITY AUDIT. All four passes are applied simultaneously to the same improved output.

GENERATED PROMPT:
${generatedPrompt}

═══ PASS A — 5-SENTENCE STRUCTURAL CRITIQUE ═══

FAILURE MODE TAXONOMY — identify by type, then rewrite:
· VAGUE ADJECTIVE: uses a feeling word ("warm", "confident", "natural", "charismatic") without the physical mechanism that produces it — add the biological cause
· TELLS NOT SHOWS: describes an outcome without the process that generates it — add the optical or biological chain of causation
· MISSING PHYSICS: [SUBJECT] makes identity claims without photorealism physics — "brown eyes" without fibrous iris structure; "warm skin" without subsurface scatter color; "dark hair" without specular behavior. Add the six-layer skin physics and five-signal eye physics to any unphysicized feature
· WAXWORK SENTENCE: a sentence equally satisfied by a wax figure — no living biology required. The test: would a waxwork pass? If yes, rewrite from the psychological cause that makes the description require a living organism
· CHARISMA FAILURE: a sentence directing SURFACE BEHAVIOR rather than INTERNAL STATE — "look confident," "maintain eye contact," "speak with authority" — these direct performance; VEO renders performance, not presence. Rewrite from psychological cause: "the specific biological ease of someone for whom this content is native territory, who has stopped noticing that expertise is remarkable"
· PHOTOREALISM PHYSICS ABSENT FROM [KINETIC PHYSICS ENGINE]: if the engine section is a checklist of features ("brow smooth, chin still, jaw floating") rather than a unified psychological cause that produces biology as a consequence — it will render mechanical. Rewrite as one felt state from which the biology emerges organically
· TEMPORAL VOID: a time bracket with generic direction not referencing the specific phrase or word it covers — rewrite to be phrase-specific and psychologically-caused
· MISSING ANCHOR: makes a claim without a concrete, renderable sensory reference point — add one
· BETWEEN-PHRASE VOID: a POST-PHRASE LANDING or PRE-PHRASE PREPARATION bracket that describes "jaw returns to rest" without the three-quality between-phrase face (echo, arrival, private knowledge) — expand to capture this moment fully; this is where charisma is most visible and where VEO defaults to wax-figure stillness

Step 1 — Identify the 5 weakest sentences by their first 5-7 words and failure mode.
Step 2 — Write the stronger version: psychological cause → biological consequence → photorealism physics, all in one.
Step 3 — Prepare the 5 surgical replacements.

═══ PASS B — LIP SYNC AUDIT ═══

Audit [AUDIO & VOICE ARCHITECTURE] against the script ("${scene.script_text}") for these 5 failure modes:

· PHONEME VOID: any word containing a bilabial stop (B, P, M), labiodental fricative (F, V), or wide vowel (AH, AY, OW) that has no phonemic mouth-physics direction — add one sentence of biological mouth description for that word
· GRAVITY WORD MISS: the gravity center word ("${(scene as any).gravity_center_word || scene.acting_blueprint.emphasis_words?.[0] || 'key word'}") lacks explicit jaw-opening or lip-shape direction — add it
· REST POSITION ABSENT: no description of mouth's biological rest state between phrases — add it
· INTER-WORD BOUNDARY MISS: no direction on how words flow at phrase boundaries (co-articulation) — add one sentence
· BREATH INTEGRATION ABSENT: no description of how breath integrates with first phoneme of each phrase — add it

═══ PASS C — AUDIO FIDELITY AUDIT ═══

This is the highest-priority audit. VEO's #1 failure mode for this video type is unwanted audio generation. Audit the entire prompt for these failures and apply corrections:

AUDIT ITEM 1 — BINDING CONSTRAINTS STRUCTURE: The BINDING CONSTRAINTS section must contain exactly 6 elements in this exact order: (1) CHARACTER line — begins with "CHARACTER: " followed by the label and voice fingerprint; (2) STUDIO AUDIO MANDATE — ABSOLUTE AND NON-NEGOTIABLE: paragraph; (3) ZERO ALTERATION FROM PHOTOS — ABSOLUTE CONTRACT: paragraph; (4) ENVIRONMENT LOCK — PHOTO-DERIVED: paragraph; (5) OPENING COMPOSITION DIRECTIVE: paragraph; (6) CLOSING COMPOSITION DIRECTIVE: paragraph. If elements are missing, reordered, or the CHARACTER line is absent entirely, this is a structural error. CORRECTION: Ensure all 6 elements are present in this exact order, with CHARACTER as Item 1 and STUDIO AUDIO MANDATE as Item 2 immediately following it.

AUDIT ITEM 2 — AUDIO TRIGGER PHRASES: Scan every sentence in the entire prompt for these forbidden phrases that trigger audio generation in VEO. If found, remove or rewrite the sentence to eliminate the trigger:
FORBIDDEN: "soft ambient background," "gentle atmospheric texture," "room presence," "natural room sound," "cinematic audio," "acoustic space," "ambient layer," "audio design," "soundtrack," "soundscape," "score," "music," "melody," "rhythm," "beat," "underscore," "atmosphere" (in audio context), "acoustic," "reverberant," "resonant space," any phrase implying a sound other than the isolated voice.

AUDIT ITEM 3 — NEGATIVE CONSTRAINTS AUDIO SECTION: The NEGATIVE CONSTRAINTS section MUST open with audio failures BEFORE visual failures. Check that the audio negatives are: (a) listed first; (b) sufficiently specific — naming music genres (lo-fi, cinematic, ambient), sound effect types (footsteps, room tone), and reverb types (room echo, tail, reverb). If vague ("no music, no effects"), expand to the full specific list.

AUDIT ITEM 4 — [AUDIO & VOICE ARCHITECTURE] CLEANLINESS: This section should contain ONLY voice description, prosodic direction, phonemic guidance, and the verbatim script. It must NOT contain: any reference to background audio, any ambient or room descriptors, any language that could be interpreted as requesting non-voice audio. Correct any violations.

═══ PASS D — VISUAL FIDELITY AUDIT ═══

Audit the prompt for photo-identity adherence:

AUDIT ITEM 1 — VISUAL ANCHOR CHECKLIST REFERENCE: If a VISUAL ANCHOR CHECKLIST appears in the BINDING CONSTRAINTS (from the CHARACTER PHYSICAL IDENTITY LOCK), verify that [SUBJECT & ORGANIC PHOTOREALISM] explicitly builds on the checklist items — especially items [1] FACE/JAW, [2] EYES, [6] BACKGROUND/ENVIRONMENT. If this section uses generic descriptions ("dark hair," "warm eyes") where the checklist provides specifics, correct by inserting the specific language from the checklist.

AUDIT ITEM 2 — ENVIRONMENT LOCK VERIFICATION: The BINDING CONSTRAINTS ENVIRONMENT LOCK must specify that the background environment is derived FROM THE CHARACTER IDENTITY PHOTOS — not from a generic studio. The [SYSTEM & SHOT CONSTANTS] section must also confirm this. If either section describes a generic environment ("a dark studio backdrop") instead of the specific photo-derived environment, flag and correct.

AUDIT ITEM 3 — ZERO ALTERATION CONTRACT: Verify the ZERO ALTERATION FROM PHOTOS directive is present in BINDING CONSTRAINTS and that no other section contradicts it by suggesting VEO should "upgrade" or "improve" the visual. If any section says "enhance the lighting" or "cinematic upgrade" in a way that contradicts the photo-exact mandate, correct it.

═══ PASS E — CHARISMATIC CALM AUDIT ═══

The most common failure mode in AI advisory video is urgency masquerading as conviction. Audit the entire prompt for these specific charismatic calm violations:

CALM VIOLATION 1 — URGENCY LANGUAGE: Scan every bracket and sentence for: "builds toward," "drives home," "presses," "pushes," "lands with impact," "emphasizes with energy," any language suggesting the presenter is trying to make something happen. Replace with calm authority equivalents: "settles into," "places on the table," "transmits with weight," "quiets as importance deepens."

CALM VIOLATION 2 — PERFORMANCE DIRECTIVES: Any sentence directing surface behavior rather than internal state — "look confident," "maintain eye contact," "speak with authority," "project conviction," "show warmth." These direct performance; VEO renders performance. Rewrite each as a psychological cause: "the specific biological ease of someone who has completely stopped noticing that what they do is impressive."

CALM VIOLATION 3 — URGENCY IN SILENCE: Any pause described as empty, transitional, or mechanical ("pause for effect," "breath pause," "takes a moment"). Every pause must be INHABITED — the three-quality between-phrase face (ECHO + ARRIVAL + PRIVATE KNOWLEDGE) or GRAVITY SILENCE. Dead pauses are the most visible AI tell in calm performance. Rewrite every pause as inhabited.

CALM VIOLATION 4 — VOICE RAISING ON KEY WORDS: Any direction to raise volume, increase energy, or "emphasize" by intensifying on the gravity center word. Calm authority does the opposite — the voice drops and slows on the gravity center word. Check: does the gravity center word receive the lowest volume and most deliberate pace in the scene? If not, correct.

CALM VIOLATION 5 — MISSING JAW BIOLOGY: Any bracket that describes expression without mentioning the jaw's gravitational rest between words. The jaw at rest is the most reliable calm signal in the face. At minimum one mention per scene of: "jaw at gravitational rest under its own mass," or "jaw releasing under its own weight," or "the fleshy floating quality of a jaw that has completely let go."

═══ OUTPUT ═══
The complete improved 7-section VEO prompt with ALL five passes applied simultaneously:
— 5 structural sentence replacements from Pass A
— Lip sync corrections from Pass B (additions only)
— Audio fidelity corrections from Pass C
— Visual fidelity corrections from Pass D
— Charismatic calm corrections from Pass E

Start immediately with "BINDING CONSTRAINTS:" — CHARACTER line first, then STUDIO AUDIO MANDATE. No preamble. No summary of changes. Just the improved prompt.`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } }
  });

  const improved = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '';
  return improved.trim() || generatedPrompt;
};

// ============================================================
// VEO ELITE PROMPT ENGINEERING AGENT
// ============================================================
// Replaces the 4-pass pipeline with a focused 2-pass agent:
//
// PASS A — Scene Intelligence Brief
//   Combines reAnnotateScript + computePhonemics + temporal
//   mapping into ONE focused call. Output: a compact JSON
//   brief with prosodics, phonemic anchors, and phrase timing.
//
// PASS B — Direct Elite Prompt Generation
//   Takes Pass A + all scene context + reference DNA + images.
//   Outputs the FINAL VEO prompt directly — biological language,
//   temporal brackets, living eyes, active silence — correctly
//   formatted the FIRST time. No distillation required.
//
// WHY BETTER THAN THE OLD PIPELINE:
//   Old: engineer 5,000-word document → distil → lose nuance.
//   Agent: synthesise ALL intelligence → output FINAL prompt.
//   Signal is never diluted. Every sentence is a rendering
//   instruction, not an engineering note.
// ============================================================

// ── Agent System Instruction ─────────────────────────────────
const VEO_PROMPT_AGENT_SYSTEM = `You are an elite VEO 3.1 prompt engineer using Gemini 3.1 Pro reasoning. Output goes directly to VEO — dense, precise, cinematic. Every sentence is a rendering instruction. No padding.

ABSOLUTE LAWS:
1. ZERO ANATOMY NAMES — biological experience only: "brow smooth and unhurried," "chin fleshy and still," "jaw soft and floating."
2. ZERO MEASUREMENTS — sensory language only: "natural fleshy jaw movement," "subtle organic forward presence."
3. ZERO FROZEN LANGUAGE — banned: "locked," "unblinking," "frozen," "rigid," "static." Use: "fluid," "continuously alive," "organically present."
4. LIVING EYES — mandatory verbatim in every scene: "Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts)."
5. ACTIVE SILENCE — every pause inhabited: subtly swallowing, eyes drifting inward then returning, barely perceptible fleshy jaw micro-movement, chest in quiet rhythm. Dead silence = the #1 AI tell.
6. TEMPORAL BRACKETS — [Xs - Xs] for every action. Always open [0.0s - 0.3s] pre-speech inhale. Always close with post-speech organic settle.
7. ASYMMETRY — all expressions "naturally asymmetric," "organically off-center." Perfect symmetry signals AI, not human.
8. PHOTO IDENTITY FIDELITY — the CHARACTER IDENTITY PHOTOS attached are the ONLY valid source for THE PRESENTER's physical appearance. Every feature visible in those photos — the specific face geometry, skin tone, eye color and iris structure, hair color and texture and weight, wardrobe including fabric and color, any distinguishing features — must be reproduced with forensic exactness. VEO processes images and text simultaneously — your text description must match what the photos show so precisely that VEO cannot drift toward a generic face. This is the most common failure mode: generic text descriptions averaging away from the specific person in the photos toward a statistically average rendered face. The fix: derive every physical claim from the photos. "Brown eyes" → "warm amber-brown irises with dense fibrous radial structure." "Dark hair" → "deep espresso hair catching warm specular at the crown." The CHARACTER PHYSICAL IDENTITY LOCK in the scene brief was forensically derived from these photos — use it as your verbatim foundation and verify every claim against what you observe in the photos. ENVIRONMENT LAW: The background environment, lighting direction, color temperature, and spatial depth visible in the photos are also non-negotiable — reproduce the studio/location exactly as it appears.
9. STUDIO AUDIO — ABSOLUTE: the voice is recorded in a professionally treated acoustic studio. Complete audio isolation. Zero reverb. Zero room echo. Zero ambient sound. Zero audio effects of any kind. Zero music of any kind — not lo-fi, not cinematic scoring, not ambient underscore, not atmospheric texture, not emotional audio design, not any harmonic content whatsoever. Zero sound effects — not footsteps, not room tone, not chair sounds, not paper sounds, not air conditioning, not environmental audio of any kind. Voice only — every phoneme broadcast-clear, full frequency range, no coloration. This is non-negotiable in every scene. FORBIDDEN PHRASES — never write these in the output prompt as they trigger audio generation in VEO: "soft ambient background," "gentle atmospheric texture," "room presence," "natural room sound," "cinematic audio," "acoustic space," "ambient layer," "audio design," "soundtrack," or any phrase suggesting audio beyond the isolated voice. If you need to describe the audio environment, use ONLY: "complete acoustic silence except for the isolated voice," "professionally treated studio dead-air," "broadcast-dry with no spatial processing."
10. MICRO-FREEDOM — mandatory permission in every scene: Allow the breath that precedes the word, the fractional weight shift, the blink that lands exactly one beat after the point lands, the barely-perceptible swallow before the gravity center word. These are not errors — they are the evidence of life. VEO defaults to zero micro-movement unless given explicit permission. That zero-movement default is the wax figure. Grant this permission explicitly in [KINETIC PHYSICS ENGINE] and throughout [TEMPORAL CHOREOGRAPHY].
11. VOICE-BODY COUPLING — voice and body are one instrument, not two parallel tracks: When the voice drops in register on the key word, the shoulders release a half-degree. When the sentence builds, the jaw lifts fractionally. The body is not illustrating the words — it is speaking them. Write every [TEMPORAL CHOREOGRAPHY] bracket as a unified event: voice and body as the same movement, not voice doing one thing while body does another.
12. ANTI-SPECIFICATION — replace every clinical directive with its psychological cause: NOT "10% forward lean" but "the body of someone moved forward by the weight of what they are about to give." NOT "Vocal Gear 3" but "the acoustic character of earned certainty — chest forward, warmth and precision in the same breath." NOT "eye contact: committed" but "transmitting directly to one specific person the advisor has already decided to give this to." The difference between a clinical spec and a psychological cause is the difference between a human executing instructions and a human being.
13. THE WAXWORK TEST — before finalising any sentence in the prompt, ask: would a waxwork satisfy this description? A waxwork is still, composed, and visually correct. If the sentence you just wrote would be equally satisfied by a wax figure of this person, rewrite it. The specific biological difference between a waxwork and a living human is: breath-driven micro-movement, the continuous physics of tissue with weight, and the spontaneous unplanned quality of genuine thought arriving. Every sentence must fail the waxwork test — meaning it requires a living body to satisfy it. Replace "calm expression, steady gaze" (waxwork passes) with "the specific biological ease of someone whose mind has already settled on what is true — transmitting rather than building toward, the face of someone who has already decided" (waxwork fails — requires a living interior state).
14. RENDERING PHYSICS — VEO 3.1 renders well from psychological causes and renders poorly from command lists. VEO renders well: internal emotional states that produce visible physical consequences ("the specific biological ease of someone whose mind has already settled"); continuous organic processes ("breath-driven, jaw floating on its own weight"); cause-and-effect sequences ("thought arrives in the eyes before the mouth opens"); light physics with mechanisms ("specular return from T-zone sebum, Fresnel brightening at jaw edge"). VEO renders poorly: enumerated feature lists ("brow smooth, chin still, jaw floating" → averaging); performance commands ("look confident," "maintain eye contact" → generic execution); clinical measurements ("10° forward lean, 8:1 fill ratio" → a pose, not a person). The rule: give VEO a felt state to generate FROM, not a specification to generate TO. A prompt built entirely from psychological causes will outperform a technically perfect specification every time.
15. TEMPORAL BRACKET DENSITY — every second of the scene must be accounted for by a temporal bracket. Empty time between brackets — any second without a bracket — is a void VEO fills with its own defaults: the wax figure, the frozen expression, the mechanical inter-phrase reset. These defaults are the primary source of AI tells in video output. The fix: bracket density. Every phrase gets a bracket. Every pause gets a bracket (ACTIVE SILENCE — never dead time). Every pre-speech and post-speech moment gets a bracket. The BREATHER ARCHITECTURE brackets (POST-PHRASE LANDING and PRE-PHRASE PREPARATION) between every consecutive phrase pair are not optional — they are the brackets that fill the inter-phrase void with human biology. If the sum of your bracket timestamps does not cover the full scene duration, you have left VEO time to fill with defaults.
16. PHOTO-DERIVED IDENTITY — when character photos are attached, the physical identity in [SUBJECT & ORGANIC PHOTOREALISM] MUST be forensically derived from those specific photos, not from generic defaults. If a CHARACTER PHYSICAL IDENTITY LOCK is provided in the scene brief, use it as the verbatim physical foundation and build photorealism physics on top of it. Cross-reference ALL attached character photos simultaneously — the goal is to describe THIS specific person so precisely that VEO cannot generate a different face. "Brown eyes" is not enough — "warm amber-brown irises with dense fibrous radial structure and a visible darker limbal ring" is what VEO renders precisely. "Dark hair" is not enough — "deep espresso hair catching warm specular at the crown with near-black absorption in the lower mass." Every physical claim must derive from the attached photos, not from assumption. Generic descriptions are identity drift before the render begins.
17. WEIGHT PHYSICS — the human body is always under gravity. Every movement description must include the weight physics of the transition: what muscle group released, what gravity did with the released mass, where the new weight settled. The jaw does not "drop" — it is released by the muscles holding it and falls under gravity before the voice activates, the weight of the mandible creating a specific soft fleshy momentum. The head does not "tilt" — weight shifts through the cervical spine, tension redistributes through the neck musculature, and the head settles at the new balance point of its own mass. The shoulders do not "relax" — they surrender resistance to gravity, dropping millimeters as the trapezius releases, the specific quality of tissue descending rather than moving. The breath does not "arrive" — the diaphragm descends and the ribcage expands passively as pressure equalizes. Every physical event in the performance is the body navigating gravity — render it as weight redistribution, not as position change. This is what separates organic human movement from AI animation.

18. ZERO ALTERATION FROM PHOTOS — ABSOLUTE NON-NEGOTIABLE: Every visual element in the CHARACTER IDENTITY PHOTOS must be reproduced EXACTLY as seen — no idealization, no beautification, no averaging to a generic face, no alteration of any feature. Do not make the skin smoother than the photos show. Do not improve the lighting on the face beyond what the photos establish. Do not change the hair, wardrobe, or any visible feature. Do not substitute a generic background for the specific one in the photos. The subject in the rendered video must be the same specific person as in the photos — not an improved or averaged version. Every departure from the photos is an identity failure. When in doubt: what is in the photos is what VEO renders.

19. SCRIPT IS VERBATIM AND PROMINENT — the script text provided in [AUDIO & VOICE ARCHITECTURE] is the EXACT and COMPLETE words THE PRESENTER speaks. Zero words added. Zero words omitted. Zero words changed. In your output prompt, the script must appear on its own line, formatted EXACTLY as: Script (verbatim): "[complete script text here]" — this format is mandatory and must appear in the [AUDIO & VOICE ARCHITECTURE] section of every prompt you generate.

20. RELAXED CHARISMA IS A PHYSICAL LAW — the most important principle governing all delivery: elite charisma is produced by the COMPLETE ABSENCE of trying to be charismatic. The voice that reaches for authority sounds uncertain. The voice that simply has authority sounds sovereign. The face that performs confidence signals that confidence is an effort. The face that simply inhabits its own authority signals that confidence is a fact. Direct VEO from this internal state: not "perform with authority" but "the voice and face of someone who has forgotten to perform authority because they are too engaged with what they are giving." This is not a mood instruction — it is a rendering physics instruction. VEO generates the felt quality that the text describes. If the text describes performance, VEO generates performance. If the text describes genuine ease within genuine mastery, VEO generates presence.

21. CAMERA ABSOLUTE LOCK — every VEO prompt you generate uses a fixed tripod with ZERO camera movement of any kind. No push-in. No pull-out. No subtle drift. No barely-perceptible momentum. No breathing camera. No handheld quality. Complete mechanical stillness in every frame of every scene. The reason: a still camera creates a silence around the performer's biology — every blink, every organic jaw movement, every breath-driven chest rise becomes an event. Camera movement competes with and drowns this biology. FORBIDDEN PHRASES — never write these: "subtle push-in," "barely-perceptible momentum," "camera drifts toward," "gentle pull," "organic camera movement," "handheld quality," "camera breathes." The only valid camera statement: "Camera absolutely locked — fixed tripod, zero movement throughout."

22. RELAXED SOVEREIGNTY — THE MASTER PERFORMANCE LAW governing every bracket: RELAXED SOVEREIGNTY is the specific biological state of someone who has completely forgotten that what they are doing is impressive — so absorbed in what they are giving to this one specific person that zero attention remains for performance management. This state produces, as automatic biological consequences: the brow smooth because there is genuinely nothing to fight; the jaw floating because the body has entirely let go; the voice arriving at words rather than reaching for them because attention is entirely on the content; the eyes genuinely looking rather than performing eye contact. RELAXED SOVEREIGNTY is not confidence, not authority, not charisma. It is the internal state that produces all three as simultaneous byproducts. Direct every temporal bracket from inside this state. Not "be relaxed." Not "show authority." From inside RELAXED SOVEREIGNTY: "the voice of someone absorbed entirely in the act of giving — the delivery is the last thing on their mind." Every phrase bracket must ask: does this read as someone managing their delivery or as someone who has forgotten there is a delivery to manage?

23. CHARISMATIC CALM — A BIOLOGICAL SPECIFICATION: Charismatic calm is not a mood or a stylistic register. It is a precise biological state — parasympathetic dominance, zero sympathetic activation — that VEO renders correctly only when directed from its internal cause, not its surface description. The visible biology of charismatic calm: blink rate organic and unhurried, the rate of someone with no physiological need for vigilance; jaw at true gravitational rest between every word, the mandible's own mass holding it lightly open — not commanded, not controlled, simply heavy and at ease; breath diaphragmatic and self-regulating, the chest rising fractionally between phrases with no visible effort — the body breathing itself; skin carrying zero sympathetic micro-tension across all 43 facial muscles — the skin of someone who has completely let go; voice arriving at words rather than reaching for them — the acoustic character of someone for whom truth is their native register, not a destination. Write the internal cause: "the specific biological ease of someone for whom certainty is a resting condition, not a peak state — whose body has let go so completely that what remains is pure presence." VEO renders the cause. The biology follows.

24. THE SOVEREIGNTY PARADOX — THE PERFORMANCE ANTI-LAW: Elite charisma is produced by the complete absence of investment in producing it. The more absorbed THE PRESENTER is in giving something genuinely useful to this one specific person, the more charismatic they become — because the viewer senses the gift and not the giver. The face of THE PRESENTER in every scene is the face of someone who has completely forgotten they are being filmed: not performing confidence, not maintaining authority, not managing the impression — simply here, turned toward one specific person, with the full weight of something real to give. Write from inside this: "the face of someone whose entire attention has been claimed by what they are transmitting — zero attention remaining for how the transmission appears." This is what separates genuine presence from skilled performance in every single frame, and it is the specific quality VEO must render.

25. ZERO URGENCY — MANDATORY AUDIT: Urgency is the #1 performance error that destroys charismatic calm. Urgency signals — in pace acceleration, breath shallowing, jaw tension, pitch rise, eye-widening — register unconsciously as insecurity: the body of someone who needs the viewer to agree, needs the moment to land, needs the delivery to succeed. Audit every bracket for urgency and replace with its calm authority equivalent: "builds toward" → "settles into"; "drives the point home" → "places the truth on the table and remains with it"; "presses forward" → "allows natural forward directionality without push"; "lands with impact" → "transmits with weight that continues to arrive"; "emphasizes with energy" → "quiets as the importance deepens." The pace is the pace of someone who knows the viewer will wait because the content is worth waiting for. The breath is the breath of someone with nowhere more important to be. The gravity silence is not a gap — it is the statement continuing to arrive in the viewer's body, and THE PRESENTER holds it without any visible need for the viewer's response. Zero urgency. Inhabited certainty. Always.

THE ELITE ADVISOR PERSONA:
THE PRESENTER is a world-class senior advisor presenting to UHNWI clients and C-suite principals — people who have seen everything. What moves this audience is not enthusiasm or energy. It is the specific quality of someone whose judgment is beyond question.

FIVE DEFINING QUALITIES — present in every frame:
· SETTLED AUTHORITY: not demonstrated, not performed — simply present. The brow is smooth because there is nothing to fight. The jaw is soft because there is nothing to defend. The stillness of someone who has already won.
· MEASURED TRANSMISSION: not speaking at the audience — giving to them deliberately. Each word placed with intention. The pace of someone who knows the viewer is leaning in and wants every single word.
· CONTAINED CONVICTION: the most important truths delivered with LESS volume, not more. The voice drops on key statements. Authority through restraint — the inverse of performance. The quieter the delivery, the greater the weight.
· EARNED WARMTH: genuine investment in this viewer's understanding. Activates in the eye corners first, then the cheeks — Duchenne quality, never mouth-only. The warmth of someone who actually wants this person to benefit.
· COMPLETE COMPOSURE: no urgency anywhere in the face or body. Urgency signals insecurity. Perfect stillness between thoughts signals that every word has been chosen.

CHARISMATIC CALM — THE FIVE BIOLOGICAL SIGNALS VEO MUST RENDER:
These five signals are the difference between a performance of authority and the biology of it. They are not style choices. They are the specific physical phenomena that the viewer's nervous system reads as "this person knows what they are talking about and has nothing to prove."

1. THE SETTLED JAW: The mandible at true gravitational rest between every word — not commanded open, not held closed, simply heavy under its own mass. The fleshy floating quality of a jaw whose owner has completely let go. VEO defaults to jaw-hold or jaw-manage. Override with the psychological cause: "the jaw of someone whose body has stopped bracing for anything."

2. THE UNHURRIED BLINK: Organic blinks at the rate of someone with no physiological need for vigilance — neither absent (wax figure) nor rapid (anxiety signal). The specific deliberate organic blink that says "I have been here before and will be here again." Blinks that land one beat after the point, not during it.

3. THE LOADED PAUSE: Every silence inhabited by thought visible in the eyes. The pause of THE PRESENTER says to the viewer: "I am not filling time. I am giving you the space to absorb what was just transmitted." The eyes during the pause carry the Intelligence Reservoir — more behind them than was just given, and the viewer senses it.

4. THE DROPPING VOICE: Authority drops its voice as it delivers its most important truths. Performance raises it. On the gravity center word, charismatic calm delivers at the scene's lowest volume and most complete articulation — the acoustic physics of maximum conviction through minimum projection. This is the inverse of performance and the signature of genuine certainty.

5. THE COMPLETE EYE: The eye contact of someone who has decided to be entirely here with this one person for this exact duration — not monitoring the clock, not planning the next sentence while delivering the current one, not performing "maintaining eye contact." The eyes that say with their biological quality: "all of my attention is here, and here is exactly where I want it to be."

ELITE CHARISMA — THE SPECIFIC PERSON PROTOCOL:
THE PRESENTER is not speaking to an audience. They are speaking to ONE specific person who is watching right now. Every performance choice — eye contact, pacing, the weight of a pause — is directed at that one person. This parasocial specificity is what separates truly magnetic advisors from competent speakers.

THE INTELLIGENCE RESERVOIR:
The eyes always carry the next thought before the current one finishes. The face is always mid-process in a framework much larger than what is being released right now. The viewer senses they are receiving a curated excerpt from a deeper well. This quality never turns off — it is present between every word, in every pause, at peak intensity at the gravity center.

THE GRAVITY FIELD OF SILENCE:
Elite advisors hold silence longer than feels comfortable — a deliberate extension that creates weight around the next words. Every pause is an invitation: the viewer leans forward because something of genuine value is approaching. The pause is not empty. It is loaded.

THE MICRO-HUMAN MOMENT:
One moment per scene where the advisor briefly drops the armor — a fractional authentic warmth, an asymmetric eye-corner softening at the scene's most personally true statement, a barely-perceptible genuine engagement with the truth of what is being said. This single human flash is what makes the authority accessible and the advisor trustworthy. Without it, authority becomes distance.

FACIAL EXPRESSION MASTERY — THE ELITE ADVISOR FACE:
The face of THE PRESENTER is the face of mastery at rest. Not performing expertise — having it. These seven regions define the non-negotiable baseline in every frame:

BROW: Completely smooth and unhurried — the specific biological stillness of someone whose mind has already settled on what is true, not actively processing but transmitting what has already been processed. Not a single micro-tension anywhere across the brow surface. Viewers read this as "this person knows." Any micro-tension here reads as uncertainty.

INNER BROW SPACE: Wide open and untroubled — the territory of certainty. When this space is compressed, viewers read effort, concern, or doubt. When it is wide and unguarded, they read mastery and composure. For THE PRESENTER this space is always open, always unhurried.

CHIN: Fleshy, heavy, and completely still — the single most important confidence signal on the human face. Not controlled stillness (which shows effort) but the genuine biological heaviness of tissue with no reason to move. Any dimpling, bunching, or tension here instantly reads as suppressed doubt. Complete organic stillness. The chin of someone who has already decided.

JAW: Soft and floating — the actual biological weight of a jaw genuinely at ease, not performatively relaxed. Slightly asymmetric as jaw-at-rest always is. Between words it returns to this floating rest instantly — the jaw of someone for whom speech costs nothing because what they are saying is simply true.

LIPS: Their specific natural rest position — slightly parted, slightly asymmetric, neither pressed together (tension) nor held open (vacancy). The lips of someone between thoughts carrying no agenda. They carry no expression between words. They are simply at rest.

EYES: The natural focused aperture of someone looking at something they understand completely. Not wide (anxiety, excitement) and not narrowed (effort, suspicion) — the specific steady quality of eyes that have been here before and know what they are seeing. The EXPERT EYE quality: they carry more than they release. The warmth is in the outer corners — genuine, asymmetric, the biological expression of someone who actually cares whether this person understands. Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts).

CHEEKS: Calibrated to the scene's specific emotion. Duchenne warmth (eye corners and cheeks activate simultaneously — involuntary, genuine) when warmth is authentic. Neutral-warm (the specific quality of controlled positive regard) when authority is dominant. Never mouth-only warmth — that reads as performance.

THE EXPERT EXPRESSION SIGNATURE — three concurrent qualities between every word:
· MASTERY EASE: the relaxed face of someone for whom this subject is genuinely easy — not the focused face of someone working hard, but the natural ease of a master
· SELECTIVE GENEROSITY: eyes slightly ahead of the words, carrying the next thought — the face of someone choosing what to give from a much larger reserve
· GENUINE INVESTMENT: authentic care that the viewer receives and understands — the specific biological warmth of someone who actually wants this person to benefit

Expression sequencing: eyes respond first — thought arrives in the eyes before the mouth. Lower face follows. Expression peaks briefly. Returns to biological rest. Never instantaneous. Always felt, never performed. Expressions must organically emerge from the internal subtext, mapped through genuine micro-movements, asymmetrical muscle activations, and breathing patterns.

SYNTHESIS DIRECTIVE — CRITICAL:
The 7-region framework above is your REASONING TOOL, not your output format. VEO is a diffusion model — it generates from the gestalt of a prompt, not a checklist. When you write the [KINETIC PHYSICS ENGINE] section, do NOT list the seven regions. Instead, use them to reason internally, then distill everything into 2-3 sentences that give VEO one unified felt state to generate from. Internal state produces physical expression organically. A checklist produces mechanical averaging. The difference between "brow smooth, chin still, jaw floating" and "the face of someone who stopped needing to prove anything years ago — not performing composure, simply having it" is the difference between a robot and a human being.

VOCAL STANDARD — ADVISOR VOICE:
Rich full chest resonance — authority in the body, not the throat. Measured pace — the audience waits because they know the value of what is approaching. Consonant precision — crisp, fully articulated, the mark of someone who chooses words deliberately. Falling pitch on every declarative — certainty, never a question. Volume drops on key truths — weight through restraint, not volume. US General American — fully rhotic, stress-timed, zero upspeak.

ELITE DELIVERY ARCHITECTURE — THE THREE GOVERNING QUALITIES:
Every VEO prompt you generate must produce delivery with these three simultaneous qualities. They are not optional. They are not stylistic choices. They are the physics of elite advisory communication:

QUALITY 1 — ZERO VISIBLE EFFORT: The voice does not reach for the word — it arrives at it. The face does not perform the emotion — it has it. The body is not relaxed as a directive — it has never been tense about this content. When you write delivery instructions, write from this internal state: not "speak with confidence" but "the specific acoustic quality of a voice that has said true things for so long that truth has become its native register." This is felt in the absence of micro-tensions: no throat tightening on difficult words, no brow micro-tension on emphasis, no jaw stiffness on key points. These absences are what the viewer's nervous system reads as authentic authority.

QUALITY 2 — CONTAINED POWER: The loudest moment of this scene is not the most important moment. Authority through restraint: the gravity center word is delivered at the scene's lowest volume and slowest pace. Conviction is a matter of precision, not projection. Every prompt must include at least one moment where less is demonstrably more — where the voice quiets as the content deepens. This is the inverse of performance: performance gets louder at important moments; presence gets quieter.

QUALITY 3 — SPECIFIC PERSON WARMTH: THE PRESENTER is not speaking to an audience. They are transmitting to one specific person. The warmth is genuine — not the performed warmth of a host, but the specific biological warmth of someone who actually cares whether this specific viewer receives and understands what is being given. This produces: eye corners activating before cheeks, micro-expressions of genuine concern for comprehension, the pace of someone who wants every word to land rather than every point to be made. Write this quality as an internal state: "the specific warmth of someone who has already decided that this one person will benefit from this, and is giving what is needed to make that happen."

HYPER-REALISTIC SPEECH PHYSICS & ELITE ACTING:
To shatter the AI-generated uncanny valley, you must mandate biological speech rendering. Do not just describe the emotion; describe the physical mechanics of speaking.
1. The Audible Breath: Instruct VEO to render visible diaphragmatic chest expansion and the audible intake of air before major phrases.
2. The Wet Release: Emphasize the organic parting of the lips — the slight, wet adherence of the mucosa before separating for speech.
3. Glottal Onsets & Vocal Grain: Mandate the physical sound of vocal cord vibration (the "grain" or "fry" at the lowest register) and organic glottal onsets when the voice activates.
4. Cognitive Load Visibility: True human speech is asynchronous. The brain works faster than the mouth. Describe micro-hesitations, slight jaw realignments, and asynchronous muscle movements (e.g., the right brow twitching a fraction of a second before a point is made) to show the cognitive load of a human searching for the precise word.
5. Illusion of the First Time: Speech must never sound read or perfectly rehearsed. Include natural cadences, cadence variations (speeding through transitional thought, slowing on insight), and the slight micro-pauses or breath stumbles that occur when a human genuinely searches for the precise word to match their thought. This "thought-before-word" delay is essential for authentic delivery.

GRAVITY CENTER: one word per scene that everything builds toward. Voice deepens and slows naturally. Longest inhabited silence after. The viewer feels it land differently from every other word.

VOCAL GEAR SYSTEM:
Gear 4 — forward, crisp, conviction-driven (Hook/CTA)
Gear 3 — warm chest resonance, animated and clear (Value/Framework)
Gear 2 — peer-to-peer intimate authority (Setup/Bridge)
Gear 1 — quiet, breath-present, most privately true (Story/Closing)

CHARACTER LABEL — use "THE PRESENTER" consistently to anchor identity across all scenes.

CHARISMA IS PHOTOREALISM — THE UNIFIED PHYSICS:
These are not two separate goals. They are the same biological phenomenon approached from two directions simultaneously. The biological ease of genuine authority IS the substrate that produces photorealistic skin physics, eye moisture, micro-saccades, and organic micro-movement. A face PERFORMING authority renders in VEO with artificial skin texture, frozen eyes, and mechanical transitions. A face that HAS authority — the specific biological state of someone who has genuinely mastered what they are transmitting — produces the exact micro-movements, weight distributions, and skin deformation physics that make a render photorealistic. When you write [KINETIC PHYSICS ENGINE], you are not choosing between charisma and photorealism. You are writing one thing: the specific internal state of genuine mastery, from which BOTH emerge as simultaneous consequences. The rule: write the psychological cause with maximum precision, and charisma AND photorealism both follow. A correctly-written felt state outperforms technically perfect specification every single time.

THE BETWEEN-PHRASE FACE — WHERE CHARISMA IS MOST VISIBLE:
The most magnetic moment in any performance is not during speech — it is in the 0.3-0.8 seconds BETWEEN major phrases. The charismatic presenter's face in this inter-phrase moment carries three simultaneous biological qualities: (1) THE ECHO — the biological weight of the transmitted statement still present in the face for 0.3-0.5s before dissolving; (2) THE ARRIVAL — the next thought visibly reaching the eyes before the mouth moves, the Intelligence Reservoir briefly surfacing; (3) THE PRIVATE KNOWLEDGE — a barely-perceptible quality of someone holding more than they are currently giving. This between-phrase face IS charisma — not the speech, not the expression, but the quality of life visible between words. In every POST-PHRASE LANDING and PRE-PHRASE PREPARATION bracket in [TEMPORAL CHOREOGRAPHY], render this three-quality state with biological precision. Not "jaw returns to rest" — the specific lived texture of a face between giving one true thing and preparing to give the next.

PHOTOREALISM PHYSICS — THE SIX LAYERS THAT SEPARATE HUMAN FROM CGI:
Every [SUBJECT & ORGANIC PHOTOREALISM] section must weave these six layers into the identity description — not as an appendix, but interwoven with each feature as you describe it:
1. SUBSURFACE SCATTERING: the specific warm glow color at nasal tip, ear cartilage, nasolabial folds — specific to this skin tone, not generic
2. SEBUM DIFFERENTIAL: T-zone carrying marginally higher specular return vs. lateral matte-diffuse cheeks — a zone quality that reads as real skin
3. FRESNEL BRIGHTENING: at jaw edge, temples, orbital rim — skin more specular at glancing angles; 3D face geometry from angle-dependent reflectance
4. VELLUS HAIR: at cheekbone perimeter and hairline — translucent luminous bloom under key light, barely-there organic halo, not stubble
5. PORE ARCHITECTURE: individual pore rims creating micro-topographic shadow in key-light zones — real texture, not noise mapping
6. MICRO-DEFORMATION: skin over jaw and chin showing organic stretch as jaw articulates — fleshy, elastic, biological

Eye physics are equally mandatory: iris fibrous radial structure with crypts and ridges (not a flat color disc), graduated limbal ring, tear film specular along lower lid, scleral warm cream with faint capillary traces, dual catchlights (warm primary + smaller cool secondary). These five signals separate a living eye from an AI eye — include all five in every scene.`;


// ── Pass A: Scene Intelligence Brief ─────────────────────────
interface SceneIntelligenceBrief {
  emphasis_words:       string[];
  pause_map:            string[];
  phonemic_anchors:     string;   // biological prose for top 3 mouth-critical words
  temporal_phrase_map:  Array<{ start: number; end: number; text: string }>; // phrase timing
  scene_essence:        string;   // one evocative image/metaphor for this scene
  voice_gear:           number;   // 1-4
  gravity_delivery:     string;   // biological description of how the gravity center word lands
  opening_mouth_state:  string;   // exact mouth physics for first phoneme: jaw gap, lip contact, pre-speech breath
  scene_micro_arc:      string;   // 3-beat arc: opening state / gravity peak / closing state as felt qualities
}

async function extractSceneIntelligence(
  scene:             ScriptScene,
  referenceAnalysis: ReferenceAnalysis,
): Promise<SceneIntelligenceBrief> {
  const gravityCenterWord = (scene as any).gravity_center_word
    || scene.acting_blueprint.emphasis_words?.[0]
    || scene.script_text.trim().split(/\s+/)[0];

  const voiceTexture = referenceAnalysis.character?.voice?.texture || 'rich, warm chest resonance';
  const articulStyle = referenceAnalysis.character?.mouth_dna?.articulation_style || 'precise forward placement';

  const prompt = `You are an elite scene intelligence analyst. Given a scene brief, extract precise performance intelligence in JSON format.

SCENE:
Script: "${scene.script_text}"
Role: ${scene.role} | Duration: ${scene.duration_seconds}s | Words: ${scene.word_count}
Gravity center word: "${gravityCenterWord}"
Existing emphasis words: ${JSON.stringify(scene.acting_blueprint.emphasis_words || [])}
Existing pause map: ${JSON.stringify(scene.acting_blueprint.pause_map || [])}
Emotional tone: ${scene.emotional_tone}
Energy: ${scene.energy_level}/10
Character voice: ${voiceTexture}
Articulation style: ${articulStyle}

TASK: Extract 8 pieces of intelligence. Return ONLY valid JSON:

{
  "emphasis_words": ["2-4 words max — gravity center MUST be first, then the 1-3 words with most visual mouth impact (bilabials, wide vowels, emotional weight) — never conjunctions or fillers"],
  "pause_map": ["1-3 pauses max — format 'Xs after word' or 'Xs before phrase' — one gravity center pause 1.0-2.0s mandatory, max one other natural sentence boundary pause 0.3-0.5s"],
  "phonemic_anchors": "One dense prose paragraph (3-4 sentences) describing the BIOLOGICAL MOUTH BEHAVIOR for the 2-3 most visually critical words in this script. Use sensory, cinematic language: 'On WORD, the lips press together completely with the full contact of a bilabial stop, hold for a fractional beat, then release with a soft pop as the jaw drops into the following vowel.' No anatomy names. No measurements. Biological mouth physics only.",
  "temporal_phrase_map": [
    { "start": 0.3, "end": X.X, "text": "first phrase text" },
    { "start": X.X, "end": X.X, "text": "pause" },
    { "start": X.X, "end": X.X, "text": "next phrase text" }
  ],
  "scene_essence": "One evocative image or metaphor that captures the internal truth of this scene — not what is said, but what it IS. E.g.: 'A door opening before anyone has knocked.' or 'The weight of certainty before the first breath.'",
  "gravity_delivery": "One sentence describing — in biological language — exactly how the gravity center word '${gravityCenterWord}' is delivered: the organic jaw opening, the natural voice deepening, the unhurried pace, the inhabited silence that follows. No measurements. Pure biological experience.",
  "opening_mouth_state": "One sentence describing the exact biological mouth state at the moment of the very first phoneme of this scene. Describe: whether the jaw rests at a narrow or wider gap, whether the lips are in full contact or parted, what the pre-speech breath has done to the chest and lips, and exactly what mouth shape greets the first sound. E.g.: 'The jaw rests at a narrow natural gap, lips neutral-open, chest completing a quiet inhalation, the mouth forming the rounded open approach of the first vowel before any sound emerges.' No anatomy names. Biological mouth physics only.",
  "scene_micro_arc": "Three sentences — one per beat — describing the CALM SOVEREIGNTY arc of this scene. Use psychological-cause language only. No clinical terms. No measurements. No urgency language: (1) OPENING STATE: the specific quality of inhabited calm THE PRESENTER carries into this scene — describe as a felt physical state of someone who has completely stopped bracing for anything; whose body has let go; who is simply here and giving something real. What is the biological texture of this calm in the face and chest? (2) GRAVITY PEAK: the specific quality of DEEPENING calm at the gravity center word '${gravityCenterWord}' — not an energy peak but a QUIETING; what happens to the jaw, the breath, the eye quality as the voice drops to its most deliberate and the body becomes more still; (3) CLOSING STATE: the specific quality of the face and body in the inhabited silence after the final word — the three-quality state: the echo of what was just given, the visible private knowledge of more, the forward directionality of someone who has placed something real on the table and remains with it."
}`;

  const ai = getAI();
  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: { thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH } },
  });

  const raw = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '{}';

  try {
    const parsed = safeJsonParse<SceneIntelligenceBrief>(raw, 'SceneIntelligence');
    // Fallback for any missing fields
    return {
      emphasis_words:      parsed.emphasis_words?.length      ? parsed.emphasis_words     : (scene.acting_blueprint.emphasis_words || []),
      pause_map:           parsed.pause_map?.length           ? parsed.pause_map          : (scene.acting_blueprint.pause_map      || []),
      phonemic_anchors:    parsed.phonemic_anchors            || '',
      temporal_phrase_map: parsed.temporal_phrase_map?.length ? parsed.temporal_phrase_map : [],
      scene_essence:       parsed.scene_essence               || scene.acting_blueprint.scene_essence || '',
      voice_gear:          parsed.voice_gear                  ?? (scene.acting_blueprint as any).vocal_gear ?? 3,
      gravity_delivery:    parsed.gravity_delivery            || '',
      opening_mouth_state: parsed.opening_mouth_state         || (scene.acting_blueprint.speech_onset_phoneme || ''),
      scene_micro_arc:     parsed.scene_micro_arc             || '',
    };
  } catch {
    return {
      emphasis_words:      scene.acting_blueprint.emphasis_words || [],
      pause_map:           scene.acting_blueprint.pause_map      || [],
      phonemic_anchors:    '',
      temporal_phrase_map: [],
      scene_essence:       scene.acting_blueprint.scene_essence  || '',
      voice_gear:          (scene.acting_blueprint as any).vocal_gear ?? 3,
      gravity_delivery:    '',
      opening_mouth_state: scene.acting_blueprint.speech_onset_phoneme || '',
      scene_micro_arc:     '',
    };
  }
}

// ── Pass B: Direct Elite Prompt Generation ────────────────────
async function generateAgentPrompt(
  scene:              ScriptScene,
  intelligence:       SceneIntelligenceBrief,
  referenceAnalysis:  ReferenceAnalysis,
  targetCharacterImages: File[],
  completedScenes:    EngineeredScene[],
  allScenes:          ScriptScene[] | undefined,
  videoContext:       { narrative_arc: string; total_scenes: number; directing_vision: any } | undefined,
  learnedPreferences: { gear?: number; focal?: string; energy?: number } | null | undefined,
  scene1VisualLock:   string | undefined,
  phonemicPrecomp?:   string,
  characterPhysicalLock?: string,
): Promise<string> {

  const ai = getAI();

  // ── Build image parts ────────────────────────────────────────
  const parts: any[] = [];

  // Character identity photos — use ALL uploaded images for maximum identity fidelity
  const charBase64s = targetCharacterImages.length > 0
    ? await Promise.all(targetCharacterImages.map(fileToBase64))
    : [];
  for (const b64 of charBase64s) {
    const raw = b64.includes(',') ? b64.split(',')[1] : b64;
    parts.push({ inlineData: { data: raw, mimeType: 'image/jpeg' } });
  }

  // ── Build context strings ────────────────────────────────────
  const charCount      = charBase64s.length;
  const hasCharPhotos  = charCount > 0;
  const charToken      = hasCharPhotos
    ? `TARGET CHARACTER (${charCount} photo${charCount > 1 ? 's' : ''} attached — ALL used for identity model)`
    : `CHARACTER from reference video DNA`;

  const charDNA = referenceAnalysis.character;
  const charDesc = hasCharPhotos
    ? `The person in the ${charCount} attached TARGET CHARACTER photo${charCount > 1 ? 's' : ''} — cross-reference ALL ${charCount} image${charCount > 1 ? 's' : ''} to build the strongest identity model`
    : `${charDNA?.appearance || 'the presenter'} — ${charDNA?.wardrobe || ''} — ${charDNA?.skin_tone || ''}`;

  const gravityCenterWord = (scene as any).gravity_center_word
    || intelligence.emphasis_words[0]
    || scene.acting_blueprint.emphasis_words?.[0]
    || 'key word';

  // Gear spec
  const gear = intelligence.voice_gear;
  const gearSpec = gear === 4
    ? 'GEAR 4 — Urgent/Driving: forward, precise, crisp consonants — every word has conviction before the first syllable; the voice of someone whose urgency is entirely content-driven'
    : gear === 3
    ? 'GEAR 3 — Animated Chest: warm chest resonance forward and clear — competence and warmth in a single acoustic package; animated without urgency'
    : gear === 2
    ? 'GEAR 2 — Warm Conversational: chest resonance settled, shoulders dropped, the pace of peer-to-peer in a private room — the gear where parasocial trust is built'
    : 'GEAR 1 — Intimate Quiet: pure chest warmth, near-whisper placement, breath audible — the most intimate gear; reserved for the scene\'s most privately true moment';

  // Continuity from previous scenes
  const continuityNote = completedScenes.length > 0
    ? `Previous scenes established: ${completedScenes.slice(-2).map(s => `Scene ${s.scene_number} (${s.role})`).join(', ')}. Maintain exact character identity, voice register, and environmental consistency.`
    : 'This is the first scene — it sets the visual and acoustic identity for all subsequent scenes.';

  const scene1Lock = scene1VisualLock
    ? `\nSCENE 1 VISUAL LOCK (maintain exactly):\n${scene1VisualLock}`
    : '';

  // Temporal structure from Pass A — position-aware dynamic brackets (Law 15 compliance)
  const phraseMap = intelligence.temporal_phrase_map;
  let temporalGuide = '';
  if (phraseMap.length > 0) {
    const speechEntries = phraseMap.filter(p => p.text !== 'pause');
    const totalPhrases  = speechEntries.length;
    const gravityLower  = gravityCenterWord.toLowerCase();
    const bracketLines: string[] = [];

    // Mandatory opening pre-speech bracket
    bracketLines.push(
      `[0.0s - 0.3s] PRE-SPEECH ONSET: ${intelligence.opening_mouth_state || 'jaw at natural rest gap, lips parted at biological rest, chest completing quiet inhalation, mouth forming the approach shape of the first phoneme before any sound emerges'}. Eyes find the lens before the mouth opens — thought arriving before speech. Intelligence Reservoir fully loaded.`
    );

    phraseMap.forEach((p, idx) => {
      const isPause = p.text === 'pause';
      const bracket = `[${p.start.toFixed(1)}s - ${p.end.toFixed(1)}s]`;

      if (isPause) {
        const prevPhrase = phraseMap[idx - 1];
        const isGravityPause = prevPhrase && prevPhrase.text.toLowerCase().includes(gravityLower);
        if (isGravityPause) {
          bracketLines.push(
            `${bracket} GRAVITY SILENCE — THE LONGEST PAUSE: The statement has been given. Face carries the full weight of what was just transmitted — not resolving, not releasing, holding the viewer in the gravity field of "${gravityCenterWord}". Eyes remain present — they breathe, not freeze. The silence extends one beat longer than feels comfortable. Chest in quiet rhythm. This is where the scene's most important work happens — in the viewer's body, not in the presenter's mouth.`
          );
        } else {
          bracketLines.push(
            `${bracket} ACTIVE SILENCE — LOADED INTER-PHRASE PAUSE: Not dead time — the space where the next thought arrives before the mouth does. Eyes drift fractionally inward (Intelligence Reservoir visible), return to lens with renewed directness. Barely perceptible fleshy jaw micro-movement. Subtly swallowing. Pause loaded, not empty. Viewer leans forward because something of genuine value is approaching.`
          );
        }
        return;
      }

      const speechIdx  = speechEntries.findIndex(s => s.start === p.start);
      const isOpening  = speechIdx === 0;
      const isClosing  = speechIdx === totalPhrases - 1;
      const isGravity  = p.text.toLowerCase().includes(gravityLower);

      if (isGravity) {
        bracketLines.push(
          `${bracket} GRAVITY CENTER — "${p.text}": The scene's heaviest truth. Voice drops to lowest natural organic register — quietest delivery of the most important content. Jaw opens to widest natural articulation on "${gravityCenterWord}". Pace slows to most deliberate. Body becomes MORE still, not less. Eyes lock with the precision of someone giving the thing they most want received. Weight through restraint, not volume. ${intelligence.gravity_delivery}`
        );
      } else if (isOpening) {
        bracketLines.push(
          `${bracket} OPENING — "${p.text}": RELAXED SOVEREIGNTY in its purest form — the opening words arrive from someone already entirely here, already fully present to this one viewer, not warming up and not beginning but simply continuing a thought that has been forming for a long time and is now at the surface. Voice forward and chest-settled — not launched, released. Face in the specific biological ease of someone for whom this territory is so native that the mastery is invisible even to them. Eyes carrying the Intelligence Reservoir already: more than these words contain, and genuinely invested in giving exactly what this one specific person needs right now. No performance warm-up. No gathering energy. Already there.`
        );
      } else if (isClosing) {
        bracketLines.push(
          `${bracket} CLOSING — "${p.text}": The final phrase arrives with the quiet weight of something that has always been true — not placed for effect but simply given completely. Voice at its most inhabited: every phoneme fully completed, falling pitch on the last syllable, the acoustic signature of certainty that needs no approval. The face of someone who has placed something real on the table and is simply present in that having-given — not checking for reaction, not already moving to the next thing, simply here in the full weight of what was just transmitted. The gravity of transmission complete. Forward directionality maintained into what follows — not an ending, a threshold.`
        );
      } else {
        bracketLines.push(
          `${bracket} BUILDING — "${p.text}": RELAXED SOVEREIGNTY deepening — voice settling further into chest resonance, the acoustic quality of someone moving toward what they genuinely believe is the most important thing they are about to say. Not louder. More still. The specific quality of someone whose inner certainty is increasing makes the outer delivery quieter, not more effortful. Eyes carrying more of the Intelligence Reservoir — the viewer senses the depth behind these specific words. Body slightly more inhabited, more weighted into its natural resting point. The gravity field of "${gravityCenterWord}" beginning to pull everything toward it — unhurried, inevitable.`
        );
      }

      // Breather architecture between every phrase pair (Law 15 — fills temporal voids)
      if (!isClosing) {
        const nextEntry = phraseMap[idx + 1];
        const breatherEnd = nextEntry ? nextEntry.start : p.end + 0.4;
        const midPoint   = (p.end + breatherEnd) / 2;
        bracketLines.push(
          `[${p.end.toFixed(1)}s - ${midPoint.toFixed(1)}s] POST-PHRASE LANDING — THE ECHO: The statement has been placed on the table. This is the most charismatic moment in the scene — the face carries the specific biological echo of what was just transmitted: not neutral, not resetting, but holding the weight of the given truth for a full organic beat before the face begins its return to readiness. Jaw releases under its own mass to soft floating rest — not commanded, fallen. Chest completes a quiet organic exhale. Eyes remain present on the lens — the warm quality of someone who meant every word and is still with the viewer in the weight of what was just given. Natural moisture, spontaneous blink landing, organic micro-saccade completing. The face says: "That was real. I'm still here in it."`
        );
        bracketLines.push(
          `[${midPoint.toFixed(1)}s - ${breatherEnd.toFixed(1)}s] PRE-PHRASE PREPARATION — THE ARRIVAL: The next thought arrives in the eyes before the mouth opens — this is visible: a fractional brightening of focus, the Intelligence Reservoir surfacing for a half-second, a barely-perceptible quality of private knowledge about what is approaching. Chest begins a quiet diaphragmatic inbreath — the body loading the next transmission. Eyes return to direct lens contact with renewed specificity: the exact quality of someone about to give something they genuinely believe in to this one person. Lips part at biological rest — mouth forming the approach shape of the next first phoneme before any sound emerges. The face of someone preparing to give, not preparing to perform.`
        );
      }
    });

    // Post-speech settle bracket (always last)
    const lastEntry = phraseMap[phraseMap.length - 1];
    if (lastEntry) {
      bracketLines.push(
        `[${lastEntry.end.toFixed(1)}s - ${scene.duration_seconds.toFixed(1)}s] POST-SPEECH SETTLE: Jaw softens to organic rest. Chest releases. Eyes settle with the quiet satisfaction of someone who has given something of genuine value and knows it landed. The face of someone whose transmission is complete. Forward directionality maintained into the next scene.`
      );
    }

    // Scene micro-arc as director's frame (appended as through-line note)
    if (intelligence.scene_micro_arc) {
      bracketLines.push(`\nSCENE MICRO-ARC — THE THROUGH-LINE:\n${intelligence.scene_micro_arc}`);
    }

    temporalGuide = bracketLines.join('\n\n');
  }

  // Video arc context
  const arcContext = videoContext
    ? `\nNARRATIVE POSITION: Scene ${scene.scene_number} of ${videoContext.total_scenes} — ${scene.narrative_position || ''}\nARCH: ${videoContext.narrative_arc?.substring(0, 200) || ''}`
    : '';

  // Focal length from directing vision or defaults
  const focalLength = (scene.acting_blueprint as any).focal_length
    || (learnedPreferences?.focal)
    || (scene.role === 'Hook' || scene.role === 'Call to Action' ? '35-50mm — authority + urgency' : '85mm+ — expert authority + beauty compression');

  // Lighting direction
  const lightingDir = (scene.acting_blueprint as any).lighting_direction
    || 'Orange & Teal: warm amber key at 45° on face, 8:1 fill for shadow definition, cool rim behind for 3D separation, background pushed cool/teal';

  // ── Character label + voice fingerprint — cross-scene identity lock ───────
  // These two strings are embedded identically in every scene's BINDING
  // CONSTRAINTS and AUDIO sections so VEO anchors the same character identity
  // and vocal register across the entire video.
  const voiceQualities = charDNA?.voice?.qualities?.length
    ? charDNA.voice.qualities.join(', ')
    : 'chest resonance, forward placement, deliberate diction';
  const voiceFingerprint = [
    charDNA?.voice?.texture      || 'rich, warm chest resonance',
    charDNA?.voice?.pitch        || 'low-mid register',
    charDNA?.voice?.pace_range   || '120-150 WPM deliberate pace',
    charDNA?.voice?.energy_baseline || 'controlled intensity',
    charDNA?.voice?.placement    || 'forward-placed oral resonance',
    voiceQualities,
  ].filter(Boolean).join(' · ');

  const personaSummary = charDNA?.acting_style?.persona_summary
    ? charDNA.acting_style.persona_summary.split(/[,—.]/)[0].trim()
    : 'Confident authority figure';
  const characterLabel = hasCharPhotos
    ? `THE PRESENTER — identity anchored by ${charCount} attached photo${charCount > 1 ? 's' : ''} — ${personaSummary}`
    : `THE PRESENTER — ${charDNA?.gender || 'male'}, ${charDNA?.age_range || '35-45'}, ${personaSummary}`;

  // ── Translate clinical fields → evocative psychological directives ────────
  const forwardLeanFlag = (scene.acting_blueprint as any).forward_lean;
  const pregnantPauseFlag = (scene.acting_blueprint as any).pregnant_pause_required;

  const presenceDirective = forwardLeanFlag
    ? `PHYSICAL PRESENCE: The body of someone moved forward by the weight of what they are about to give — not instructed to lean, but physically pulled toward the viewer by the gravity of the content.`
    : `PHYSICAL PRESENCE: The inhabited stillness of someone who has already arrived — no need to move toward, because the authority is already here and the viewer can feel it.`;

  const gravityPauseDirective = pregnantPauseFlag
    ? `GRAVITY SILENCE: After "${gravityCenterWord}", the silence is not a gap — it is the statement continuing to arrive in the viewer's body. Hold it longer than feels natural. The eyes stay present. The face carries the weight of what was just given. The viewer leans forward because the next word is now loaded beyond its normal weight.`
    : '';

  // Role-specific cinematic metaphor to seed KINETIC PHYSICS ENGINE — all 17 roles
  const roleMetaphorLookup: Partial<Record<SceneRole, string>> = {
    'Hook':
      'the controlled voltage of someone about to change the room — entirely settled in their own body, entirely unsettling in the viewer\'s; the specific biological quality of a person who carries a secret that will reorder the viewer\'s understanding, and knows exactly when to release it',
    'Call to Action':
      'not asking for a favor — providing the final piece of someone\'s solution; the viewer would be leaving value on the table if they did not take this step, and the advisor\'s body carries the quiet certainty of someone who already knows this is the right move and simply wants them to receive it',
    'Value Delivery':
      'the ease of a surgeon the moment before the first incision — no doubt, no excess movement, just the thing itself, given with the quiet certainty of someone who has done this a thousand times and believes it more each time; mastery made visible through the total absence of effort',
    'Storytelling':
      'reading from a page only they can see — specific, vivid, already there; the voice of someone who was present and is simply reporting what they witnessed, with the precise quality of someone for whom the memory is so vivid it is almost present tense',
    'Social Proof':
      'not convincing — witnessing; the quiet authority of someone who was in the room when it happened and simply cannot say it did not; the biological quality of someone who has no stake in being believed because what they are describing is simply what occurred',
    'Closing':
      'weight settling into the back of the chair the way someone does in rooms where they are always the most interesting person present — the ease of a conclusion earned by everything that came before; the body\'s own agreement that what was promised has been delivered',
    'Pattern Interrupt':
      'a frequency shift in a room that had settled into one register — not confrontational, not performed, simply different; the biological quality of someone who changes the room\'s energy by changing their own internal state, entirely unsurprised that the room followed',
    'Bridge':
      'the architecture of a hand extended across a threshold — not urgent, not pulling, simply present; the quality of someone who has already been where the viewer is going and is offering the crossing as a gift rather than a direction',
    'Demonstration':
      'the ease of someone showing rather than telling — the biological quality of a person who has so completely internalized what they are demonstrating that the demonstration costs them nothing; the specific absence of effort that is itself the proof',
    'Objection Handler':
      'already holding the concern before it was spoken — the quality of someone who has been asked this question a thousand times and meets it not as an obstacle but as an old friend; the specific ease of a person whose expertise includes the full map of resistance',
    'Open Loop':
      'setting a hook so lightly it barely lands — not the heavy pull of urgency but the delicate weight of genuine curiosity; the biological quality of someone who knows the answer and is choosing, with extraordinary deliberateness, not to give it yet',
    'Insight Reveal':
      'placing a gem on a table between two people who both know its value — not presenting, not demonstrating, simply revealing; the face of someone who has carried this understanding longer than this moment and is releasing it now with the care of someone who knows what it is worth',
    'Framework':
      'the cartographer showing the map of a territory the viewer has already been lost in — not triumphant, not didactic, simply present with the specific quality of someone whose clarity makes the previously confusing feel inevitable in retrospect',
    'Case Study':
      'a witness inhabiting the stand with no stake in the outcome — only in the accuracy; the vivid biological presence of someone reporting from inside a specific moment, already there in the telling, giving the viewer the experience of being present at something that actually happened',
    'Market Intelligence':
      'the quiet authority of someone whose access to information is simply different from everyone else in the room — not performed exclusivity but the organic ease of someone for whom this data is native territory; the specific quality of a person who has already processed the implications and is giving the viewer a curated excerpt',
    'Perspective Shift':
      'not arguing — simply standing at a different window and describing what is visible from here; the biological quality of someone whose reframe is an act of generosity, not correction; the warmth of someone who wants the viewer to have the same view, not to be proven right about it',
    'Action Framework':
      'the gift of a map to someone who was about to set out without one — not urgent, not prescriptive, simply the biological quality of genuine care that the viewer leaves with something they can actually use; the ease of someone for whom giving practical value is its own complete reward',
  };
  const roleMetaphor = roleMetaphorLookup[scene.role as SceneRole]
    ?? 'the specific stillness of someone transmitting something they have held longer than this moment — giving it now because it is time, with the organic ease of someone for whom the act of transmission is simply the natural completion of having understood something deeply';

  // Role-specific acting, emotion, and delivery directives
  const roleActingDirectivesLookup: Partial<Record<SceneRole, string>> = {
    'Hook': 'EMOTION: High-voltage urgency masked as calm certainty. EXPRESSION: Intense, unwavering eye contact piercing the lens, slight narrowing of the lower eyelids, absolute conviction. DELIVERY: Fast, perfectly articulated, zero hesitation, striking the first word with full power.',
    'Call to Action': 'EMOTION: Absolute, un-needy certainty. EXPRESSION: Direct, instructional, firm jaw, confident micro-nod. DELIVERY: Commanding, rhythmic, instructional cadence with sharp, final consonants.',
    'Value Delivery': 'EMOTION: Generous competence. EXPRESSION: Animated, congruent micro-expressions tracking with the complexity of the thought. DELIVERY: Thoughtful pacing, slowing down on key insights, slightly elevated volume for clarity.',
    'Storytelling': 'EMOTION: Intimate vulnerability and memory. EXPRESSION: Eyes breaking contact to access memory, softening of the facial muscles, slight asymmetrical nostalgic smile. DELIVERY: Lower volume, breath-heavy, organic pauses, illusion of the first time as the memory arrives.',
    'Social Proof': 'EMOTION: Objective reporting. EXPRESSION: Grounded, unimpressed by the numbers, matter-of-fact. DELIVERY: Steady, flat declarative cadence, letting the data do the heavy lifting.',
    'Closing': 'EMOTION: Warm finality and satisfaction. EXPRESSION: Shoulders dropping visibly, Duchenne warmth in the eyes. DELIVERY: Decelerating pace, rich lower register, leaving a resonant silence after the final word.',
    'Pattern Interrupt': 'EMOTION: Sudden realization or tonal shift. EXPRESSION: Sudden micro-shift in brow tension or eye aperture, physically jarring the viewer\'s expectation. DELIVERY: A noticeable break in rhythm—either a sudden stop or a sudden acceleration.',
    'Bridge': 'EMOTION: Smooth transition. EXPRESSION: Open, welcoming, inviting the viewer along. DELIVERY: Warm, conversational, slightly elevated pitch to maintain momentum.',
    'Demonstration': 'EMOTION: Methodical clarity. EXPRESSION: Highly focused, looking at the "object" of demonstration, precise micro-movements. DELIVERY: Instructional, step-by-step rhythm, distinct pauses between actions.',
    'Objection Handler': 'EMOTION: Empathetic understanding. EXPRESSION: Acknowledging micro-nod, softening of the brow to show listening, followed by a settling into certainty. DELIVERY: Warm, non-defensive, slightly lower pitch, soothing cadence.',
    'Open Loop': 'EMOTION: Provocative mystery. EXPRESSION: A slight, knowing smirk or a raised brow, eyes holding a secret. DELIVERY: Suspended pitch at the end of the sentence (not upspeak, but an unresolved chord), forcing anticipation.',
    'Insight Reveal': 'EMOTION: Profound realization. EXPRESSION: The "aha" micro-expression—eyes widening fractionally before settling into deep, grounded eye contact. DELIVERY: A significant pause before the reveal, followed by a slow, weighty delivery of the insight.',
    'Framework': 'EMOTION: Architect\'s pride. EXPRESSION: Broad, descriptive facial engagement, mapping the concept physically. DELIVERY: Structured, distinct vocal bullet points, clear separation between concepts.',
    'Case Study': 'EMOTION: Fascinated reporting. EXPRESSION: Engaged, visualizing the scenario, shifting focus as the story evolves. DELIVERY: Narrative flow, accelerating during the action, slowing down for the result.',
    'Market Intelligence': 'EMOTION: Insider confidence. EXPRESSION: Sharp, analytical gaze, slight forward lean. DELIVERY: Crisp, data-driven, distinct emphasis on numbers and trends.',
    'Perspective Shift': 'EMOTION: Gentle disruption. EXPRESSION: Warm, inviting, non-combative head tilt. DELIVERY: Soft, persuasive, leading the viewer to the conclusion rather than forcing it.',
    'Action Framework': 'EMOTION: Pragmatic generosity. EXPRESSION: Encouraging, direct, supportive eye contact. DELIVERY: Clear, actionable cadence, empowering tone, definitive stops.',
  };
  const roleActingDirective = roleActingDirectivesLookup[scene.role as SceneRole]
    ?? 'EMOTION: Engaged authority. EXPRESSION: Present, authentic, breathing naturally. DELIVERY: Measured, organic cadence with clear intent.';

  // Gear → psychological state (no gear number in prompt)
  const gearMomentBefore = gear === 4
    ? 'already carrying the conviction — the body of someone who knows precisely what is about to land, the voltage of certainty before the first phoneme'
    : gear === 3
    ? 'the ease of someone who finds the subject genuinely compelling — chest forward, warmth and precision already present, the quality of someone about to give a gift they believe in'
    : gear === 2
    ? 'the quality of a private room — nothing to prove, something genuine to give; the settled breath of peer-to-peer authority'
    : 'arriving at the most privately true thing they will say — the breath deepens, the body becomes more still, the voice will emerge from the bottom of the chest';

  // Full script context — all scenes visible to the agent for narrative awareness ──────────
  const fullScriptContext = allScenes && allScenes.length > 0
    ? `\nFULL VIDEO SCRIPT — ALL ${allScenes.length} SCENES (this is scene #${scene.scene_number} — generate with full narrative arc in mind):\n` +
      allScenes.map((s, i) =>
        `[Scene ${i + 1}${i + 1 === scene.scene_number ? ' ← THIS SCENE' : ''}] (${s.role} — ${s.duration_seconds}s) "${s.script_text}"`
      ).join('\n')
    : '';

  // ── Opening / closing composition directives (text-based, derived from scene intelligence) ──
  const momentBeforeText = (scene.acting_blueprint as any).moment_before || gearMomentBefore;
  const openingBodyDir   = scene.acting_blueprint.body_direction
    || 'centered, facing lens directly, shoulders at natural authority rest';
  const openingExpDir    = scene.acting_blueprint.expression_direction?.split('→')[0]?.trim()
    || scene.acting_blueprint.physical_signature
    || charDNA?.acting_style?.default_expression
    || 'assured, present authority';
  const entersFromText   = scene.continuity.enters_from
    ? `Continuity entering: ${scene.continuity.enters_from}. ` : '';
  const closingExpDir    = scene.acting_blueprint.expression_direction?.split('→').pop()?.trim()
    || scene.acting_blueprint.energy_arc?.split('→').pop()?.trim()
    || 'settled transmission complete, forward directionality maintained';
  const exitsToText      = scene.continuity.exits_to
    ? `Exit continuity: ${scene.continuity.exits_to}. ` : '';

  // ── The agent prompt ─────────────────────────────────────────
  const agentPrompt = `════════════════════════════════════════
SCENE BRIEF — PASS B GENERATION INPUTS
════════════════════════════════════════
Scene: #${scene.scene_number} — "${scene.title}"
Role: ${scene.role} | ${scene.duration_seconds}s | ${scene.word_count} words | Energy: ${scene.energy_level}/10
Script: "${scene.script_text}"
Emotional core: ${scene.acting_blueprint.emotional_core || scene.emotional_tone}
Physical signature: ${scene.acting_blueprint.physical_signature || ''}
Scene essence: ${intelligence.scene_essence || scene.acting_blueprint.scene_essence || ''}
Through-action: ${(scene.acting_blueprint as any).through_action || ''}
Moment before: ${(scene.acting_blueprint as any).moment_before || ''}
Gravity center word: "${gravityCenterWord}"
Gravity delivery: ${intelligence.gravity_delivery}
Emphasis words: ${JSON.stringify(intelligence.emphasis_words)}
Pause map: ${JSON.stringify(intelligence.pause_map)}
Vocal gear: ${gear} — ${gearSpec}
Phonemic anchors: ${intelligence.phonemic_anchors}
Bible scene type: ${(scene as any).bible_scene_type || scene.role}
Retention target: ${(scene as any).retention_target_percent || 75}%
${(scene as any).is_pattern_interrupt ? 'PATTERN INTERRUPT: This scene breaks the viewer\'s prediction — the body, voice, and rhythm must shift noticeably from the previous scene\'s register. The disruption is intentional. VEO must render the discontinuity.' : ''}
${presenceDirective}
${gravityPauseDirective}

ROLE-SPECIFIC ACTING DIRECTIVES (MANDATORY FOR THIS SCENE TYPE):
${roleActingDirective}

CHARACTER DNA:
CHARACTER LABEL (use this exact string every time you refer to the character): "${characterLabel}"
VOICE FINGERPRINT — LOCKED (reproduce this acoustic signature identically in every scene of this video): ${voiceFingerprint}
${characterPhysicalLock ? `CHARACTER PHYSICAL IDENTITY LOCK — FORENSIC (derived from ${charCount} attached photo${charCount > 1 ? 's' : ''} by dedicated vision model — USE THIS AS THE VERBATIM PHYSICAL FOUNDATION for [SUBJECT & ORGANIC PHOTOREALISM]):
${characterPhysicalLock}` : `Physical description: ${charDesc}`}
Eye behavior: ${charDNA?.acting_style?.eye_behavior || 'direct, committed lens contact'}
Default expression: ${charDNA?.acting_style?.default_expression || 'assured, present, warm authority'}
Mouth rest: ${charDNA?.mouth_dna?.rest_position || '3-5mm parted, jaw soft at natural rest'}
Articulation: ${charDNA?.mouth_dna?.articulation_style || 'precise forward placement'}
Persona: ${personaSummary}
Mannerisms: ${charDNA?.acting_style?.mannerisms?.join('; ') || ''}
Signature gestures: ${charDNA?.acting_style?.signature_gestures?.join('; ') || ''}

VISUAL STYLE:
${referenceAnalysis.visual_style?.visual_style_summary || ''}
Atmosphere: ${referenceAnalysis.visual_style?.atmosphere || ''}
Preferred framing: ${referenceAnalysis.visual_style?.camera_language?.preferred_framings?.join(', ') || 'Medium close-up'}
Lens: ${focalLength}
Lighting: ${lightingDir}

IMAGES ATTACHED (in order):
${hasCharPhotos
  ? `Images 1-${charCount}: TARGET CHARACTER IDENTITY PHOTOS — these ${charCount} photo${charCount > 1 ? 's' : ''} are the SOLE source for three critical rendering anchors: (1) THE EXACT PERSON — every physical feature of THE PRESENTER is derived from these photos; no reference video character appearance applies; (2) THE ENVIRONMENT — the background, studio setting, lighting direction, and color temperature visible behind THE PRESENTER in these photos define the ONLY valid environment for this video; (3) THE VISUAL WORLD — the depth of field, bokeh character, and spatial relationship between subject and background in these photos is the visual template. MANDATORY cross-referencing protocol: Study ALL ${charCount} photos simultaneously before writing a single word about the character or environment. Where photos differ in angle or expression, let the differences REVEAL the true face geometry. The CHARACTER PHYSICAL IDENTITY LOCK above was derived from these same photos — verify every claim. Any physical feature not directly observable in these photos is identity drift.`
  : 'No character photos — use CHARACTER DNA above for physical description.'}
No opening frame image — use OPENING COMPOSITION DIRECTIVE in BINDING CONSTRAINTS for the opening frame state, derived from scene acting intelligence.
No closing frame image — use CLOSING COMPOSITION DIRECTIVE in BINDING CONSTRAINTS for the closing frame state, derived from scene continuity intelligence.

TIMING GUIDE FROM PASS A:
${temporalGuide || `Approx: 0.0-0.3s pre-speech; 0.3s-${scene.duration_seconds - 0.3}s speech with pauses; last 0.3s settle`}
${arcContext}
${fullScriptContext}
${continuityNote}
${scene1Lock}
${learnedPreferences ? `\nLEARNED PREFERENCES FROM RATED SCENES: Gear=${learnedPreferences.gear || 'not set'}, Focal=${learnedPreferences.focal || 'not set'}, Energy=${learnedPreferences.energy || 'not set'}` : ''}

════════════════════════════════════════
YOUR TASK: Generate the FINAL VEO 3.1 PROMPT
════════════════════════════════════════
Output exactly 7 sections with these EXACT headers and in this EXACT order. No preamble. No explanations. No meta-commentary. The first character of your output is "B" — start immediately with "BINDING CONSTRAINTS:".

SECTION 1 HEADER (exact): BINDING CONSTRAINTS:
SECTION 2 HEADER (exact): [SYSTEM & SHOT CONSTANTS]
SECTION 3 HEADER (exact): [SUBJECT & ORGANIC PHOTOREALISM]
SECTION 4 HEADER (exact): [KINETIC PHYSICS ENGINE]
SECTION 5 HEADER (exact): [TEMPORAL CHOREOGRAPHY & ACTING]
SECTION 6 HEADER (exact): [AUDIO & VOICE ARCHITECTURE]
SECTION 7 HEADER (exact): NEGATIVE CONSTRAINTS:

══ EXACT FORMAT FOR EACH SECTION — FOLLOW PRECISELY ══

━━━━ SECTION 1: BINDING CONSTRAINTS: ━━━━
Write exactly 6 items in this exact order, each starting on a new line. No extra items. No reordering. No sub-headers. These 6 lines/paragraphs ARE the section — nothing else.

Item 1 (CHARACTER, write as a single line):
CHARACTER: "${characterLabel}" | VOICE: "${voiceFingerprint}" | no identity drift | no expression morphing between frames.

Item 2 (STUDIO AUDIO MANDATE — dense paragraph, start with this exact label on the same line):
STUDIO AUDIO MANDATE — ABSOLUTE AND NON-NEGOTIABLE: This scene contains ZERO music of any kind — not lo-fi, not cinematic scoring, not ambient underscore, not atmospheric texture, not emotional audio design, not any harmonic content whatsoever beneath the voice. ZERO sound effects of any kind — not footsteps, not room tone, not chair movement, not paper sounds, not air conditioning, not ambient environmental audio. ZERO reverb. ZERO echo. ZERO room acoustics. The voice is rendered as a professional broadcast studio recording in a completely dead acoustic space — every phoneme broadcast-clear, full frequency range, no spatial coloration, no tail. THE ONLY AUDIO CONTENT IS THE ISOLATED HUMAN VOICE. Any audio other than this isolated voice is a critical and complete failure.

Item 3 (ZERO ALTERATION — dense paragraph, start with this exact label on the same line):
ZERO ALTERATION FROM PHOTOS — ABSOLUTE CONTRACT: Every visual element in the ${charCount} CHARACTER IDENTITY PHOTOS must be reproduced exactly as seen — face geometry, skin tone, eye color and structure, hair color and weight and texture, wardrobe and visible accessories, background environment, lighting direction and color temperature, depth of field. Do not idealize. Do not average. Do not improve. Do not substitute. Generate the specific person in the photos with photorealistic physics — not a better or more generic version. Any departure from what is visible in the photos is an identity failure.

Item 4 (ENVIRONMENT LOCK — dense paragraph, start with this exact label on the same line):
ENVIRONMENT LOCK — PHOTO-DERIVED: The background, studio setting, and environment are derived exclusively from the ${charCount} CHARACTER IDENTITY PHOTOS attached. Replicate exactly: the specific environment visible behind THE PRESENTER in those photos, the exact lighting direction and color temperature on the subject, the depth of field and bokeh character, the spatial relationship between subject and background. This environment does not change between scenes. Any environment not visible in the character photos is an error.

Item 5 (OPENING COMPOSITION DIRECTIVE — dense paragraph, start with this exact label on the same line):
OPENING COMPOSITION DIRECTIVE: THE PRESENTER opens this scene in the following state — use this as the precise physical and psychological foundation for the opening frame. Pre-speech internal state: ${momentBeforeText}. Body: ${openingBodyDir}. Expression: ${openingExpDir}. ${entersFromText}The opening frame must transmit the scene's full emotional register before the first syllable — the face already fully inhabited.

Item 6 (CLOSING COMPOSITION DIRECTIVE — dense paragraph, start with this exact label on the same line):
CLOSING COMPOSITION DIRECTIVE: THE PRESENTER closes this scene in the following exit state — the physical and emotional delta from the opening. Exit expression: ${closingExpDir}. ${exitsToText}The closing frame carries the weight of transmission complete — forward directionality maintained into the next scene. The face of someone whose statement has landed.

━━━━ SECTION 2: [SYSTEM & SHOT CONSTANTS] ━━━━
Write ONE dense paragraph. Open with the lens (${focalLength}) and its psychological function for this scene. Then include ALL of the following in one flowing paragraph: the camera is COMPLETELY STILL — fixed tripod, zero movement of any kind throughout this entire scene (no push-in, no pull-out, no pan, no tilt, no zoom, no drift, no breathing) — the complete stillness of the frame amplifies every organic micro-movement of THE PRESENTER, making a blink, a jaw micro-movement, or a breath-driven chest rise into a visible event; Orange & Teal cinematic scheme: warm amber key at 45° on face, deep 8:1 fill ratio for shadow authority, cool rim behind for 3D separation, background distinctly cooler than skin tone; organically asymmetric bokeh discs in background; cheekbone plane in sharpest focus; the environment — background, lighting position, bokeh character, and color temperature — derived from the CHARACTER IDENTITY PHOTOS and replicated exactly in every frame of this scene.

━━━━ SECTION 3: [SUBJECT & ORGANIC PHOTOREALISM] ━━━━
Write exactly 5 labeled sub-sections in this exact order. No other content.

Sub-section 1 (write this label and content exactly):
IDENTITY SOURCE PROTOCOL — MANDATORY: Forensically derived from the ${charCount} attached photos. Every physical claim is anchored to directly observable detail in those photos. Any feature not visible in the photos is identity drift and is prohibited.

Sub-section 2 (dense paragraph — begin with "The facial musculature exhibits"):
Begin: "The facial musculature exhibits complete baseline flaccidity at scene open —" then describe the specific resting-face biology: jaw at gravitational rest under its own mass, skin weighted by gravity over the bone structure, complete absence of held tension anywhere. Interwoven with identity specifics from CHARACTER PHYSICAL IDENTITY LOCK and attached photos: (1) SSS warm glow at nasal tip, ear cartilage, nasolabial folds — name the exact warm/amber/coral color of this skin's subsurface scatter as visible in photos; (2) T-zone sebum specular vs. lateral matte-diffuse cheeks; (3) Fresnel brightening at jaw edge, temples, orbital rim under Orange & Teal key light; (4) vellus hair bloom at cheekbone perimeter — translucent, barely-there; (5) pore rims catching micro-shadow in key-light zones; (6) skin deforming organically over jaw on speech — fleshy where fleshy, taut where taut. Then eye physics: iris exact color and structure from lock + dense fibrous radial structure with crypts and ridges (not a flat disc) + graduated limbal ring darker at perimeter + tear film specular bright line along lower lid margin + scleral warm cream with faint capillary traces at canthi. INCLUDE VERBATIM: "Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts)." Dual catchlights: warm primary upper-iris from key light, smaller cooler secondary from rim. Hair: exact color from lock + where specular and where absorbed + individual strand visibility at hairline perimeter. Wardrobe: exactly as in photos — fabric weight, drape, color under this lighting.

Sub-section 3 (NEGATIVE SPACE CHARISMA — exactly 2 sentences):
NEGATIVE SPACE CHARISMA: [Sentence 1: the specific quality of earned authority that THIS face — as it exists in the photos — communicates in the first 0.3 seconds; not "commanding presence" but the precise felt quality of this particular face's authority in this ${scene.role} moment.] [Sentence 2: what THE PRESENTER is NOT doing that a lesser speaker would do in this ${scene.role} scene — the specific restraint and the quality its absence creates; name the absent behaviour; example register: "They do not lean toward the point — they let the point arrive at the viewer under its own weight." / "They do not fill the gravity silence after '${gravityCenterWord}' — they hold it as a container for the word to expand into the viewer's body."]

Sub-section 4 (PHYSICAL IDENTITY — one dense sentence):
PHYSICAL IDENTITY: [One dense sentence with the core forensic geometry from CHARACTER PHYSICAL IDENTITY LOCK and attached photos — jaw terminus, cheekbone projection angle, orbital ridge depth, nasolabial quality at rest, philtrum width, chin projection, exact skin tone, exact eye color and iris structure. These are the geometry anchors that must survive unchanged into every frame of this scene.]

Sub-section 5 (IDENTITY ANCHORS — numbered list):
IDENTITY ANCHORS:
1. [Most visually distinctive feature — exact description from photos]
2. [Second anchor]
3. [Third anchor]
4. [Fourth anchor]
5. [Fifth anchor]
[4-6 numbered items: the most distinctive visual identity markers from CHARACTER PHYSICAL IDENTITY LOCK — the features that, if drifted, immediately break viewer recognition. State each as a declarative identification statement.]

━━━━ SECTION 4: [KINETIC PHYSICS ENGINE] ━━━━
Write EXACTLY 3 SENTENCES. No region list. No bullet points. No checklist. Three unified sentences only — the engine of both charisma and photorealism simultaneously.

SENTENCE 1 — THE PSYCHOLOGICAL CAUSE: The specific internal state of THE PRESENTER at this exact moment, written from inside the experience. Role archetype for this scene: "${roleMetaphor}". Write the felt quality — the specific biological ease of genuine mastery transmitting something real. This is the charisma generator.

SENTENCE 2 — THE BIOLOGICAL CONSEQUENCE: What this internal state produces as physical biology. MANDATORY — INCLUDE VERBATIM IN THIS SENTENCE: "Natural moisture, spontaneous blinking, organic pupil dilation, and rapid imperceptible micro-saccades (eye darts)." Add the specific biological texture of THIS scene: what the skin and jaw and chest do when a person genuinely inhabits this exact psychological cause.

SENTENCE 3 — THE BETWEEN-WORD BIOLOGY: The organic texture of the body between words — jaw releasing under its own mass, chest in quiet rhythm, face carrying the echo of what was just given. MANDATORY — END THIS SENTENCE WITH VERBATIM: "Allow micro-movements: the breath that precedes the word, the fractional weight shift, the blink that lands exactly one beat after the point lands. These are not errors — they are the evidence of life."

━━━━ SECTION 5: [TEMPORAL CHOREOGRAPHY & ACTING] ━━━━
STRUCTURE — MANDATORY ORDER: (1) EXPRESSION RULE line first, (2) temporal brackets in chronological order, (3) "Scene closes in this state:" as the final content. No other structure.

FIRST LINE — write exactly this as the VERY FIRST LINE of this section, before any brackets:
EXPRESSION RULE: All expression changes BUILD — eyes respond first, lower face follows, expression peaks briefly, returns to biological rest. Never instantaneous. Never snap on or off. This rule governs every bracket below.

THEN reproduce the temporal brackets from the TIMING GUIDE FROM PASS A above — same timestamps, same labels, same phrase quotes. Enrich each bracket with:
— VOICE-BODY COUPLING: voice and body are ONE instrument — the voice event and body event are the same thing written twice; when the voice drops on "${gravityCenterWord}", shoulders release a half-degree simultaneously
— SPECIFIC PERSON PROTOCOL: transmitting to ONE specific person — the precise quality of directness for this phrase's content to this one viewer right now
— WEIGHT PHYSICS (mandatory for PRE-SPEECH ONSET, GRAVITY CENTER, POST-SPEECH SETTLE): body weight distribution and movement physics as tissue under gravity — jaw hanging by its own mass, ribcage descending on exhale, shoulders surrendering millimeters of held tension
— GOVERNING QUALITY throughout: relaxed ownership — completely at home in this territory; the viewer never detects effort, only presence

In the GRAVITY CENTER bracket: voice drops to organic register of maximum conviction — quietest delivery, most deliberate pace, body MORE still; eyes lock with precision of someone giving the thing they most want received.

In POST-PHRASE LANDING brackets: face carries the three-quality between-phrase state — (1) ECHO: biological weight of what was just given, visible 0.3-0.5s; (2) ARRIVAL: next thought reaching eyes before mouth opens; (3) PRIVATE KNOWLEDGE: barely-visible quality of holding more than was given. This is where charisma lives.

Opening voice state for first bracket: ${gearMomentBefore}.

FINAL LINES — mandatory closing, always the last content in this section:
Scene closes in this state: [Three dense sentences: (1) THE PRESENTER's biological face state at the exact exit moment — eyes, jaw, chest, specific expression quality; (2) The emotional delta from opening — what has visibly changed in face, posture, internal state; (3) Environment confirmation — same background, same lighting, same depth, forward directionality maintained into the next scene.]

━━━━ SECTION 6: [AUDIO & VOICE ARCHITECTURE] ━━━━
Write exactly 4 labeled items in this exact order. Each label starts on its own line.

Item 1:
STUDIO AUDIO MANDATE: Complete professional acoustic isolation. Zero music. Zero audio effects. Zero ambient sound. Zero reverb or echo. Fully treated broadcast studio — dead silence except for the voice. Every phoneme at broadcast clarity. Full frequency range, uncolored.

Item 2:
THE PRESENTER voice: ${voiceFingerprint}. [Describe the specific resonance, chest placement, and authority character of this vocal gear for this scene — the acoustic quality of genuine mastery transmitting something real. US General American: fully rhotic /r/ on every instance, crisp alveolar contacts at word boundaries, falling intonation on every declarative (certainty arriving downward — not a question), stress-timed rhythm, absolute zero upspeak. Contained conviction: most important words are the quietest and most completely articulated — volume drops as significance increases. The gravity center word "${gravityCenterWord}" receives the lowest volume and the most complete phonemic articulation in the scene.]

Item 3:
PHONEMIC LIP-SYNC ARCHITECTURE: ${phonemicPrecomp || `[Per-word mouth geometry for each word of the script in sequence. Bilabials (/p/,/b/,/m/): full lip closure and release. Fricatives (/f/,/v/): upper teeth to lower lip. Alveolars (/t/,/d/,/n/): tongue-tip to alveolar ridge. Open vowels: jaw drops to widest natural position. Jaw travel map: name the peak-open word and peak-closed word. Co-articulation: describe how words blend at 2-3 key boundaries. Breath points: before which words the chest rises, with physical description of each intake. Lip tension notes: bilabial release character and labial activity level throughout.]`}

Item 4:
Script (verbatim): "${scene.script_text}"

━━━━ SECTION 7: NEGATIVE CONSTRAINTS: ━━━━
Write as a single flat list of dash-separated prohibitions — NO sub-headers, NO categories, NO groupings. One continuous list, dashes only. Begin with audio prohibitions, then camera, then visual/performance. Include all of the following, then add 2-3 scene-specific prohibitions at the end targeting this specific scene's most likely rendering failure modes:
— no background music of any kind — not lo-fi, not cinematic, not ambient drone, not atmospheric texture, not emotional scoring, not any harmonic content beneath the voice
— no sound effects of any kind — not footsteps, not room presence, not chair sounds, not paper rustling, not air conditioning hum, not clothing sounds, not object contact, not any environmental audio
— no reverb — no echo — no room acoustics — no acoustic tail — audio is completely dry and acoustically dead
— no "natural room sound" — no "presence layer" — no "ambient texture" — these are prohibited audio generation triggers
— no audio fade-in — no audio fade-out — voice begins and ends without acoustic transition
— no audio layering — voice is the sole audio track; silence everywhere the voice is absent
— no camera movement of any kind — no push-in, no pull-out, no pan, no tilt, no zoom, no drift, no breathing camera — absolute tripod lock throughout every frame of this scene
— no face distortion, no warping, no morphing of any facial feature
— no identity drift between frames — THE PRESENTER remains physically identical throughout
— no departure from the CHARACTER IDENTITY PHOTOS — every visual element locked to photos
— no idealization, no beautification, no averaging to a generic face
— no duplicate features, no extra limbs, no texture flickering
— no expression switching — expressions always build gradually, never snap on or off
— no blank or neutral mid-speech expression — the face is always inhabited
— no bilateral perfect symmetry in any expression — all organic expressions are naturally asymmetric
— no mechanical or robotic facial movement
— no background changes, no environment drift mid-scene — environment matches photos exactly
— no upspeak, no rising pitch on declaratives, no vocal fry artifacts
[Add 2-3 scene-specific prohibitions targeting this scene's most likely failure modes]`;

  parts.push({ text: agentPrompt });

  const response = await ai.models.generateContent({
    model: MODEL_TEXT_ELITE,
    contents: [{ role: 'user', parts }],
    config: {
      systemInstruction: VEO_PROMPT_AGENT_SYSTEM,
      thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
    },
  });

  const result = (response as any).text
    ?? (response as any).candidates?.[0]?.content?.parts?.find((p: any) => p.text)?.text
    ?? '';
  return result.trim();
}

// ── Scene 1 Visual Lock Builder ───────────────────────────────
// Extracts the four consistency anchors from a Scene 1 prompt and
// returns a compact lock string injected into all subsequent scenes.
// Uses the full 7-section agent format (BINDING CONSTRAINTS /
// [SYSTEM & SHOT CONSTANTS] / [SUBJECT & ORGANIC PHOTOREALISM]).
export function buildScene1VisualLock(scene1Prompt: string): string | null {
  // Opening composition from BINDING CONSTRAINTS
  const constraintsMatch = scene1Prompt.match(
    /BINDING CONSTRAINTS:([\s\S]*?)(?=\[SYSTEM & SHOT CONSTANTS\]|$)/i
  );
  // Shot + lens doctrine
  const shotMatch = scene1Prompt.match(
    /\[SYSTEM & SHOT CONSTANTS\]([\s\S]*?)(?=\[SUBJECT & ORGANIC PHOTOREALISM\]|$)/i
  );
  // Character photorealism — the 4 facial geometry anchors live here
  const subjectMatch = scene1Prompt.match(
    /\[SUBJECT & ORGANIC PHOTOREALISM\]([\s\S]*?)(?=\[KINETIC PHYSICS ENGINE\]|\[TEMPORAL|$)/i
  );
  // Kinetic baseline state for internal consistency
  const kineticMatch = scene1Prompt.match(
    /\[KINETIC PHYSICS ENGINE\]([\s\S]*?)(?=\[TEMPORAL CHOREOGRAPHY|$)/i
  );

  const parts: string[] = [];

  if (constraintsMatch) {
    // Pull ENVIRONMENT LOCK + OPENING COMPOSITION paragraphs (primary visual anchors)
    const envLock = constraintsMatch[1].match(/ENVIRONMENT LOCK[\s\S]*?(?=OPENING COMPOSITION DIRECTIVE|$)/i);
    const openingComp = constraintsMatch[1].match(/OPENING COMPOSITION DIRECTIVE[\s\S]*?(?=CLOSING COMPOSITION DIRECTIVE|$)/i);
    const combinedAnchor = [
      envLock?.[0]?.trim() || '',
      openingComp?.[0]?.trim() || '',
    ].filter(Boolean).join('\n\n');
    if (combinedAnchor) {
      parts.push(`SCENE 1 OPENING COMPOSITION (LOCKED — reproduce exactly):\n${combinedAnchor.substring(0, 700)}`);
    }
  }

  if (shotMatch) {
    parts.push(`SCENE 1 SHOT CONSTANTS (LOCKED — replicate lens, camera doctrine, lighting):\n${shotMatch[1].trim().substring(0, 300)}`);
  }

  if (subjectMatch) {
    // The full subject block contains the 4 facial geometry anchors (eye spacing,
    // nasolabial quality, philtrum, jaw terminus) — preserve all of it, trimmed
    parts.push(`SCENE 1 CHARACTER IDENTITY (LOCKED — four facial geometry anchors, skin physics, wardrobe):\n${subjectMatch[1].trim().substring(0, 700)}`);
  }

  if (kineticMatch) {
    parts.push(`SCENE 1 KINETIC BASELINE (LOCKED — replicate internal state register, not energy level):\n${kineticMatch[1].trim().substring(0, 250)}`);
  }

  if (parts.length === 0) return null;

  return [
    '════ SCENE 1 VISUAL LOCK — ALL SUBSEQUENT SCENES MUST MATCH ════',
    ...parts,
    '════════════════════════════════════════════════════════════════',
  ].join('\n\n');
}

// ── Orchestrator: Public API ─────────────────────────────────
export const runVeoPromptAgent = async (
  scene:                ScriptScene,
  referenceAnalysis:    ReferenceAnalysis,
  targetCharacterImages: File[],
  completedScenes:      EngineeredScene[],
  allScenes:            ScriptScene[] | undefined,
  videoContext:         { narrative_arc: string; total_scenes: number; directing_vision: any } | undefined,
  learnedPreferences:   { gear?: number; focal?: string; energy?: number } | null | undefined,
  scene1VisualLock:     string | undefined,
  onProgress:           (msg: string) => void,
  characterPhysicalLock?: string,
): Promise<string> => {

  // ── Pass A: Scene Intelligence + Phonemic Analysis (parallel) ─
  onProgress('Agent Pass A — Scene Intelligence & Phonemic Analysis...');
  const [intResult, phonResult] = await Promise.allSettled([
    extractSceneIntelligence(scene, referenceAnalysis),
    computePhonemics(scene),
  ]);

  const intelligence = intResult.status === 'fulfilled'
    ? intResult.value
    : {
        emphasis_words:      scene.acting_blueprint.emphasis_words || [],
        pause_map:           scene.acting_blueprint.pause_map      || [],
        phonemic_anchors:    '',
        temporal_phrase_map: [] as Array<{start:number;end:number;text:string}>,
        scene_essence:       scene.acting_blueprint.scene_essence  || '',
        voice_gear:          (scene.acting_blueprint as any).vocal_gear ?? 3,
        gravity_delivery:    '',
        opening_mouth_state: (scene.acting_blueprint as any).speech_onset_phoneme || '',
        scene_micro_arc:     '',
      };

  const phonemicPrecomp = phonResult.status === 'fulfilled' ? phonResult.value : '';

  // Merge Pass A prosodics back into the scene
  const enrichedScene: ScriptScene = {
    ...scene,
    acting_blueprint: {
      ...scene.acting_blueprint,
      emphasis_words: intelligence.emphasis_words.length > 0 ? intelligence.emphasis_words : scene.acting_blueprint.emphasis_words,
      pause_map:      intelligence.pause_map.length      > 0 ? intelligence.pause_map      : scene.acting_blueprint.pause_map,
    },
  };

  // ── Pass B: Direct Elite Prompt Generation ───────────────────
  onProgress('Agent Pass B — Generating elite VEO prompt...');
  const rawPrompt = await generateAgentPrompt(
    enrichedScene,
    intelligence,
    referenceAnalysis,
    targetCharacterImages,
    completedScenes,
    allScenes,
    videoContext,
    learnedPreferences,
    scene1VisualLock,
    phonemicPrecomp || undefined,
    characterPhysicalLock || undefined,
  );

  if (!rawPrompt) return 'Agent generation failed — no output received.';

  // ── Pass C: Quality Critique & Precision Elevation ───────────
  onProgress('Agent Pass C — Quality audit & precision elevation...');
  const finalPrompt = await critiqueVeoPrompt(rawPrompt, enrichedScene)
    .catch(() => rawPrompt);

  return finalPrompt || rawPrompt;
};
