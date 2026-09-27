/**
 * laya-skill-router: pre-loads relevant skills into pi's context using a
 * locally-served convaiinnovations/laya System-1 decision model.
 *
 * Every turn, before the agent loop starts, all known skills are scored by
 * the laya sidecar (http://127.0.0.1:7699). The top-k "core" skills are
 * injected as a visible custom message containing their verbatim SKILL.md
 * bodies, so the main model never has to self-trigger the read.
 *
 * Install: add this file's path to settings.json "extensions".
 * Sidecar: uv run sidecar/server.py  (see README.md)
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { readFileSync, appendFileSync, mkdirSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join, basename, dirname } from "node:path";

const ROUTE_URL = process.env.LAYA_ROUTER_URL ?? "http://127.0.0.1:7699/route";
const MODE = (process.env.LAYA_ROUTER_MODE ?? "observe") as "observe" | "inject";
// Defaults = the measured operating point (docs/ROUTER_LATENCY_STRATEGY.md):
// threshold 0.4 + descriptions in the question gave recall@3 0.736 / prec@3
// 0.302 / no-load picks 1.91 through the harness, vs 0.623 / 0.270 / 3.73 for
// the old 0.3 bare-question full-catalog config.
const THRESHOLD = Number(process.env.LAYA_ROUTER_THRESHOLD ?? "0.4");
const TOP_K = Number(process.env.LAYA_ROUTER_TOP_K ?? "3");
const MAX_SKILL_CHARS = Number(process.env.LAYA_ROUTER_MAX_SKILL_CHARS ?? "8000");
const FETCH_TIMEOUT_MS = Number(process.env.LAYA_ROUTER_TIMEOUT_MS ?? "3500");
const BLOCK_BUDGET_MS = Number(process.env.LAYA_ROUTER_BLOCK_BUDGET_MS ?? "450");
const USE_DESC = (process.env.LAYA_ROUTER_USE_DESC ?? "1") === "1";
const LOG_DIR = join(homedir(), ".pi", "agent", "laya-router");
const LOG_FILE = join(LOG_DIR, "log.jsonl");

const GLOBAL_SKILL_DIRS = [
  join(homedir(), ".agents", "skills"),
  join(homedir(), ".pi", "agent", "skills"),
];

interface RoutedSkill {
  name: string;
  description: string;
  path: string;
}

interface RouteResponse {
  picks: { name: string; p: number }[];
  all: { name: string; p: number }[];
  latency_ms: number;
  path?: string; // full | shortlist | shortlist-fallback | cache
  cached?: boolean;
  s1_ms?: number;
  s2_ms?: number;
  n_scored?: number;
}

function log(entry: Record<string, unknown>): void {
  try {
    mkdirSync(LOG_DIR, { recursive: true });
    appendFileSync(LOG_FILE, JSON.stringify({ ts: new Date().toISOString(), ...entry }) + "\n");
  } catch {
    /* logging must never break a turn */
  }
}

function parseFrontmatterSkill(skillMdPath: string): RoutedSkill | null {
  try {
    const text = readFileSync(skillMdPath, "utf8");
    if (!text.startsWith("---")) return null;
    const end = text.indexOf("---", 3);
    if (end === -1) return null;
    const fm = text.slice(3, end);
    const nameMatch = fm.match(/^name:\s*(.+)$/m);
    const descMatch = fm.match(/^description:\s*(.+)$/m);
    return {
      name: (nameMatch?.[1] ?? basename(dirname(skillMdPath))).trim(),
      description: (descMatch?.[1] ?? "").trim().slice(0, 300),
      path: skillMdPath,
    };
  } catch {
    return null;
  }
}

function scanSkillDirs(cwd: string): RoutedSkill[] {
  const dirs = [...GLOBAL_SKILL_DIRS, join(cwd, ".agents", "skills"), join(cwd, ".claude", "skills")];
  const seen = new Set<string>();
  const skills: RoutedSkill[] = [];
  for (const dir of dirs) {
    let names: string[] = [];
    try {
      names = readdirSync(dir);
    } catch {
      continue;
    }
    for (const entry of names) {
      const skillMd = join(dir, entry, "SKILL.md");
      const parsed = parseFrontmatterSkill(skillMd);
      if (parsed && !seen.has(parsed.name)) {
        seen.add(parsed.name);
        skills.push(parsed);
      }
    }
  }
  return skills;
}

/** Normalize pi's own skill list (shape may vary); fall back to dir scan. */
function normalizeSkills(raw: unknown, cwd: string): { skills: RoutedSkill[]; source: string } {
  if (Array.isArray(raw) && raw.length > 0) {
    const skills: RoutedSkill[] = [];
    for (const item of raw) {
      if (!item || typeof item !== "object") continue;
      const rec = item as Record<string, unknown>;
      const path = String(rec.path ?? rec.filePath ?? rec.file ?? "");
      const name = String(rec.name ?? rec.skill ?? (path ? basename(dirname(path)) : ""));
      if (!name) continue;
      skills.push({
        name,
        description: String(rec.description ?? "").slice(0, 300),
        path: path || join("(unknown)", name, "SKILL.md"),
      });
    }
    if (skills.length > 0) return { skills, source: "systemPromptOptions" };
  }
  return { skills: scanSkillDirs(cwd), source: "dir-scan" };
}

function stripFrontmatter(text: string): string {
  if (!text.startsWith("---")) return text;
  const end = text.indexOf("---", 3);
  return end === -1 ? text : text.slice(end + 3).replace(/^\s+/, "");
}

export default function layaSkillRouter(pi: ExtensionAPI) {
  const injectedThisSession = new Set<string>();
  let sessionEnabled: boolean | null = null; // null = not yet health-checked
  let lastPrompt = "";
  let lastSkills: RoutedSkill[] | null = null; // for input-time prefetch
  let activeCtx: ExtensionContext | null = null; // for late-delivery status updates
  let routeGeneration = 0; // newer turns invalidate older late routes
  let pendingRoute: {
    gen: number;
    state: string;
    skillsFp: string;
    promise: Promise<RouteResponse>;
  } | null = null;

  const skillsFp = (skills: RoutedSkill[]): string => skills.map((s) => s.name).join("\x1f");

  // Same state the sidecar sees, computed identically in `input` and
  // `before_agent_start` so the prefetch is reusable.
  const computeState = (prompt: string): string =>
    prompt.trim().length < 60 && lastPrompt ? `${lastPrompt.slice(0, 600)} || ${prompt}` : prompt;

  const doRoute = (state: string, skills: RoutedSkill[]): Promise<RouteResponse> =>
    fetch(ROUTE_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        state: state.slice(0, 1200),
        skills: skills.map((s) => ({ name: s.name, description: s.description })),
        threshold: THRESHOLD,
        top_k: TOP_K,
        use_desc: USE_DESC,
      }),
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    }).then((res) => {
      if (!res.ok) throw new Error(`sidecar ${res.status}`);
      return res.json() as Promise<RouteResponse>;
    });

  /** Fire the route without awaiting it; failures land in the log. */
  const startRoute = (state: string, skills: RoutedSkill[], gen: number): void => {
    const promise = doRoute(state, skills);
    promise.catch((e) => log({ event: "error", error: String(e) }));
    pendingRoute = { gen, state, skillsFp: skillsFp(skills), promise };
  };

  pi.on("session_start", async (_event, ctx) => {
    injectedThisSession.clear();
    sessionEnabled = null;
    pendingRoute = null;
    routeGeneration++;
    try {
      const res = await fetch(ROUTE_URL.replace(/\/route$/, "/health"), { signal: AbortSignal.timeout(400) });
      sessionEnabled = res.ok;
    } catch {
      sessionEnabled = false;
    }
    if (!sessionEnabled) {
      ctx.ui.setStatus("laya-router", "sidecar down (start: uv run sidecar/server.py)");
    }
  });

  // The fetch starts the instant the user submits; `before_agent_start`
  // only waits for it (bounded). See docs/ROUTER_LATENCY_STRATEGY.md.
  pi.on("input", async (event, ctx) => {
    if (sessionEnabled === false) return;
    const prompt = (event.text ?? "").trim();
    if (!prompt || prompt.startsWith("/")) return;
    const skills = lastSkills ?? normalizeSkills(undefined, ctx.cwd).skills;
    if (skills.length === 0) return;
    const state = computeState(prompt);
    lastPrompt = prompt; // before_agent_start recomputes the same state
    startRoute(state, skills, ++routeGeneration);
  });

  pi.on("session_compact", async () => {
    // compacted transcripts may drop earlier injections; allow re-injection
    injectedThisSession.clear();
  });

  /** Shared post-route work: filter fresh picks, read bodies, log. */
  const resolvePicks = (
    r: RouteResponse,
    skills: RoutedSkill[],
    source: string,
    state: string,
    late: boolean,
  ) => {
    const picked = r.picks ?? [];
    const fresh = picked.filter((p) => !injectedThisSession.has(p.name));
    const byName = new Map(skills.map((s) => [s.name, s]));

    let body = "";
    let totalChars = 0;
    const loaded: string[] = [];
    for (const p of fresh) {
      const skill = byName.get(p.name);
      if (!skill?.path || !skill.path.endsWith("SKILL.md")) continue;
      let text: string;
      try {
        text = stripFrontmatter(readFileSync(skill.path, "utf8"));
      } catch {
        continue;
      }
      if (totalChars + text.length > MAX_SKILL_CHARS * TOP_K) break;
      totalChars += text.length;
      loaded.push(p.name);
      body += `\n## skill: ${p.name} (laya p=${p.p.toFixed(2)})\n\n${text}\n`;
      injectedThisSession.add(p.name);
    }

    const already = picked.filter((p) => injectedThisSession.has(p.name) && !loaded.includes(p.name));
    log({
      event: "route",
      prompt_head: state.slice(0, 100),
      skill_source: source,
      n_skills: skills.length,
      picks: picked.map((p) => ({ name: p.name, p: Number(p.p.toFixed(3)) })),
      loaded,
      already_in_context: already.map((p) => p.name),
      sidecar_ms: r.latency_ms,
      path: r.path,
      cached: r.cached ?? false,
      s1_ms: r.s1_ms,
      s2_ms: r.s2_ms,
      delivered_late: late,
    });
    return { picked, loaded, already, body };
  };

  const buildMessage = (body: string) => ({
    customType: "laya-skill-router",
    content: `<laya-loaded-skills>\nThe laya skill router pre-loaded the following skills for this request. Their full instructions follow; use them without needing to read the files.\n${body}</laya-loaded-skills>`,
    display: true,
  });

  /** Late route (past the block budget): steer the active run instead of
   *  blocking it. Mid-run, pi delivers this as a steer (the model sees it at
   *  its next request); between turns it becomes persistent context. */
  const deliverLate = (r: RouteResponse, skills: RoutedSkill[], gen: number, state: string, source: string) => {
    if (gen !== routeGeneration) return; // a newer turn owns routing now
    const { picked, loaded, body } = resolvePicks(r, skills, source, state, true);
    const summary = `late ${r.latency_ms}ms via ${r.path ?? "full"}: ${picked.map((p) => `${p.name} (${p.p.toFixed(2)})`).join(", ")}`;
    if (MODE !== "inject") {
      try {
        activeCtx?.ui.setStatus("laya-router", `observe ${summary}`);
      } catch { /* renderer may be gone; the log has it */ }
      return;
    }
    if (loaded.length === 0) {
      try {
        activeCtx?.ui.setStatus("laya-router", summary);
      } catch { /* ignore */ }
      return;
    }
    pi.sendMessage(buildMessage(body));
    try {
      activeCtx?.ui.setStatus("laya-router", `steered: ${loaded.join(", ")} (${r.latency_ms}ms, ${r.path ?? "full"})`);
    } catch { /* ignore */ }
  };

  pi.on("before_agent_start", async (event, ctx) => {
    if (sessionEnabled === false) return;
    const prompt = event.prompt ?? "";
    if (!prompt.trim() || prompt.trimStart().startsWith("/")) {
      lastPrompt = prompt || lastPrompt;
      return;
    }

    const opts = (event as unknown as { systemPromptOptions?: { skills?: unknown } }).systemPromptOptions;
    const { skills, source } = normalizeSkills(opts?.skills, ctx.cwd);
    if (skills.length === 0) return;
    lastSkills = skills;
    activeCtx = ctx;

    // short continuations carry no signal on their own; include the prior turn
    const state = computeState(prompt);
    lastPrompt = prompt;
    const gen = ++routeGeneration;

    // In observe mode the route only feeds the footer + log — blocking the
    // turn for that is pure waste. Budget 0: the result lands late via the
    // steer path, turn cost stays at zero. Inject mode keeps the real budget
    // because first-request injection is worth paying for.
    const budget = MODE === "observe" ? 0 : BLOCK_BUDGET_MS;

    // Reuse the input-time prefetch when it matches this turn; else start now.
    if (!pendingRoute || pendingRoute.state !== state || pendingRoute.skillsFp !== skillsFp(skills)) {
      startRoute(state, skills, gen);
    }
    const pr = pendingRoute!;

    const raced = await Promise.race([
      pr.promise.then(
        (v) => ["done", v] as const,
        (e) => ["error", e] as const,
      ),
      new Promise<"timeout">((resolve) => setTimeout(() => resolve("timeout"), budget)),
    ]);

    if (raced === "timeout") {
      // Budget spent: return with 0 further turn cost. The late result
      // steers the active run (or lands as context) instead of being lost —
      // the old behavior stalled up to 3.5s and then injected nothing.
      pr.promise.then(
        (r) => deliverLate(r, skills, gen, state, source),
        () => {}, // already logged in startRoute
      );
      try {
        ctx.ui.setStatus("laya-router", `routing in background (budget ${budget}ms)…`);
      } catch { /* ignore */ }
      return;
    }
    if (raced[0] === "error") {
      log({ event: "error", error: String(raced[1]) });
      return; // fail open: turn proceeds without injection
    }

    const r = raced[1];
    const { picked, loaded, already, body } = resolvePicks(r, skills, source, state, false);

    if (loaded.length === 0 && already.length === 0) {
      ctx.ui.setStatus("laya-router", `no relevant skills (${r.latency_ms}ms)`);
      return;
    }

    if (MODE === "observe") {
      ctx.ui.setStatus(
        "laya-router",
        `observe: ${picked.map((p) => `${p.name} (${p.p.toFixed(2)})`).join(", ")} (${r.latency_ms}ms)`,
      );
      return;
    }

    const summary = [
      `laya skill router: ${picked.map((p) => `${p.name} (${p.p.toFixed(2)})`).join(", ")}`,
      loaded.length > 0 ? `loaded now: ${loaded.join(", ")}` : null,
      already.length > 0 ? `already in context: ${already.map((p) => p.name).join(", ")}` : null,
    ]
      .filter(Boolean)
      .join(" | ");

    ctx.ui.setStatus("laya-router", summary);

    if (loaded.length === 0) {
      // decision ran, picks were all injected earlier; nothing new to send
      return;
    }

    return { message: buildMessage(body) };
  });

  pi.registerCommand("routerstats", {
    description: "Show laya skill-router decision stats from the log",
    handler: async (_args, ctx) => {
      let lines: string[] = [];
      try {
        const text = readFileSync(LOG_FILE, "utf8");
        lines = text.trim().split("\n").slice(-500);
      } catch {
        ctx.ui.notify("No laya router log yet.", "info");
        return;
      }
      const entries = lines.map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean) as Record<string, unknown>[];
      const routes = entries.filter((e) => e.event === "route") as Array<Record<string, unknown>>;
      if (routes.length === 0) {
        ctx.ui.notify("No routing decisions logged yet.", "info");
        return;
      }
      const errors = entries.filter((e) => e.event === "error").length;
      const picks: Record<string, number> = {};
      let totalPicks = 0;
      let totalSidecarMs = 0;
      for (const r of routes) {
        totalSidecarMs += Number(r.sidecar_ms ?? 0);
        for (const p of (r.picks ?? []) as Array<{ name: string }>) {
          picks[p.name] = (picks[p.name] ?? 0) + 1;
          totalPicks++;
        }
      }
      const top = Object.entries(picks).sort((a, b) => b[1] - a[1]).slice(0, 10);
      const msg = [
        `laya router: ${routes.length} decisions (${errors} errors), avg sidecar ${Math.round(totalSidecarMs / routes.length)}ms`,
        `avg picks/turn: ${(totalPicks / routes.length).toFixed(2)}`,
        "top picked skills:",
        ...top.map(([n, c]) => `  ${String(c).padStart(3)}  ${n}`),
      ].join("\n");
      ctx.ui.notify(msg, "info");
    },
  });
}
