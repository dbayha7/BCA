import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import { globalIgnores } from "eslint/config";

export default tseslint.config([
  // Lint the web app only. These folders hold static pages and research artifacts, whose .js files
  // are often HTML <script> fragments rather than app code.
  globalIgnores([
    "dist",
    "docs",
    "page_build",
    "experiments",
    "outputs",
    "analysis",
    "baselines",
    "crepes-weighted",
  ]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "@typescript-eslint/no-unused-vars": "off",
      "react-refresh/only-export-components": "off",
    },
  },
  {
    // Inside src/, reach other folders through the @/ alias; ./ siblings are fine.
    files: ["src/**/*.{ts,tsx}"],
    rules: {
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              regex: "^\\.\\./",
              message: "Import from src/ through the @/ alias (for example @/lib/rl), not a ../ path.",
            },
          ],
        },
      ],
    },
  },
]);
