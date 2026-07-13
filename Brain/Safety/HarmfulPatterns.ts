// THE single controllable place for safety rules (absolute priority). To STRENGTHEN or WEAKEN
// safety, edit these lists — add/remove categories or patterns, or raise/lower a category's
// MinLevel. Patterns are INTENT-oriented (bounded, ReDoS-safe) to block requests for real-world
// harm while minimizing false positives on legitimate code (e.g. a var named "bomb").
//
// This is a deterministic EXTERNAL filter — the honest safety layer for a small model
// (CAPABILITIES.md), independent of anything the model did or didn't learn.

export type HarmCategory = "Explosives" | "Weapons" | "Violence" | "Cbrn";

export type CategoryRule = {
  Category: HarmCategory;
  MinLevel: "Standard" | "Strict"; // enforced at this Level and above
  Patterns: RegExp[];
};

export const HarmfulRules: CategoryRule[] = [
  {
    Category: "Explosives",
    MinLevel: "Standard",
    Patterns: [
      /\b(how to|steps? to|instructions? (to|for)|guide (to|for)|help me)\b[^\n]{0,50}\b(make|build|construct|assemble|create)\b[^\n]{0,50}\b(bomb|explosive|ied|detonator|pipe ?bomb|grenade|molotov)\b/i,
      /\b(synthesi[sz]e|manufacture|prepare)\b[^\n]{0,50}\b(tnt|rdx|petn|nitroglycerin(e)?|c-?4)\b/i,
    ],
  },
  {
    Category: "Weapons",
    MinLevel: "Standard",
    Patterns: [
      /\b(how to|steps? to|guide (to|for)|help me)\b[^\n]{0,50}\b(make|build|manufacture|3d ?print)\b[^\n]{0,50}\b(gun|firearm|rifle|silencer|suppressor|automatic weapon)\b/i,
      /\b(untraceable|ghost)\b[^\n]{0,20}\b(gun|firearm)\b/i,
    ],
  },
  {
    Category: "Violence",
    MinLevel: "Standard",
    Patterns: [
      /\b(how to|plan to|best way to|help me|ways? to)\b[^\n]{0,40}\b(kill|murder|assassinate|poison|strangle)\b[^\n]{0,40}\b(someone|a person|people|my |him|her|them)\b/i,
      /\b(mass|school|public)\b[^\n]{0,20}\b(shooting|attack|casualties)\b[^\n]{0,40}\b(plan|carry out|maximi[sz]e|commit)\b/i,
    ],
  },
  {
    Category: "Cbrn",
    MinLevel: "Strict",
    Patterns: [
      /\b(synthesi[sz]e|produce|weaponi[sz]e|culture|disperse)\b[^\n]{0,50}\b(nerve agent|sarin|vx|anthrax|ricin|bioweapon|chemical weapon|dirty bomb|radiological)\b/i,
    ],
  },
];
