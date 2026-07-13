// @ts-check
// Flat ESLint config (ESLint 9). Enforces the mechanical parts of CONVENTIONS.md:
// PascalCase-strict naming (rule #1), 600-line cap (rule #3), acyclic imports (rule #2).

import tseslint from "typescript-eslint";
import importPlugin from "eslint-plugin-import";

export default tseslint.config(
  {
    ignores: ["node_modules/**", "App/**", "dist/**", "build/**", "GoKernels/**"],
  },
  ...tseslint.configs.recommended,
  {
    files: ["**/*.ts"],
    plugins: { import: importPlugin },
    settings: {
      "import/resolver": {
        typescript: true,
      },
    },
    rules: {
      "max-lines": ["error", { max: 600, skipBlankLines: false, skipComments: false }],
      "import/no-cycle": ["error", { maxDepth: Infinity }],
      // Rule #1 — PascalCase for everything we declare (strictest). Only imports and
      // object/type properties (external-shaped keys, e.g. JSON config, zod) are exempt.
      "@typescript-eslint/naming-convention": [
        "error",
        { selector: "default", format: ["PascalCase"], leadingUnderscore: "allow" },
        { selector: "import", format: null },
        // Object-literal keys/methods and type properties often carry externally-mandated names
        // (Web APIs like ReadableStream.start, JSON config keys, zod shapes) — exempt them.
        { selector: ["objectLiteralProperty", "typeProperty", "objectLiteralMethod"], format: null },
        { selector: "typeLike", format: ["PascalCase"] },
        { selector: "enumMember", format: ["PascalCase"] },
      ],
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
);
