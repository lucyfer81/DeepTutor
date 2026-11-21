import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const readWebFile = (...parts: string[]) =>
  readFileSync(path.join(process.cwd(), ...parts), "utf8");

const HARNESSES = [
  { kind: "grok", route: "grok", label: "Grok CLI" },
  { kind: "hermes", route: "hermes", label: "Hermes Agent" },
  { kind: "openclaw", route: "openclaw", label: "OpenClaw" },
  {
    kind: "deepseek_harness",
    route: "deepseek-harness",
    label: "DeepSeek Harness",
  },
  { kind: "pi", route: "pi", label: "Pi Coding Agent" },
] as const;

describe("subagent harnesses", () => {
  it("new harnesses are present in connected-agent labels and glyph dispatch", () => {
    const icons = readWebFile("components", "agents", "agent-icons.tsx");
    // The names and the glyphs live side by side, shared by the agents page and
    // the composer's menus.
    const labels = icons.slice(icons.indexOf("export function agentKindLabel"));

    for (const harness of HARNESSES) {
      expect(labels).toMatch(new RegExp(`kind === ["']${harness.kind}["']`));
      expect(labels).toMatch(new RegExp(harness.label));
      expect(icons).toMatch(new RegExp(`kind === ["']${harness.kind}["']`));
    }
  });

  it("Gemini CLI is retired and agent glyphs use official local assets", () => {
    const connected = readWebFile("components", "agents", "ConnectedAgents.tsx");
    const icons = readWebFile("components", "agents", "agent-icons.tsx");
    const registry = readWebFile(
      "..",
      "deeptutor",
      "services",
      "subagent",
      "registry.py",
    );

    expect(connected).not.toMatch(/kind === ["']gemini["']/);
    expect(registry).not.toMatch(/GeminiBackend/);
    const retiredRoute = path.join(
      process.cwd(),
      "app/(utility)/settings/agents/gemini/page.tsx",
    );
    expect(existsSync(retiredRoute)).toBe(false);

    for (const asset of [
      "kimi.svg",
      "opencode.svg",
      "mimo-code.svg",
      "hermes.svg",
      "openclaw.svg",
      "deepseek-harness.svg",
    ]) {
      expect(
        existsSync(path.join(process.cwd(), "public", "agent-icons", asset)),
        `${asset} official icon asset is missing`,
      ).toBe(true);
      expect(icons).toMatch(new RegExp(`/agent-icons/${asset}`));
    }
  });

  it("new harnesses have settings routes and independent editors", () => {
    const editor = readWebFile(
      "components",
      "settings",
      "SubagentSettingsEditor.tsx",
    );
    const category = readWebFile(
      "components",
      "settings",
      "SettingsPageContent.tsx",
    );
    const nav = readWebFile(
      "features",
      "settings",
      "navigation",
      "settings-nav.ts",
    );

    for (const harness of HARNESSES) {
      expect(editor).toMatch(new RegExp(`${harness.kind}: \\{`));
      expect(category).toMatch(new RegExp(`agent-${harness.route}`));
      expect(nav).toMatch(new RegExp(`agent-${harness.route}`));
      expect(nav).toMatch(new RegExp(`/settings#agent-${harness.route}`));
    }
  });
});
