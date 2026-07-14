import { test, expect } from "bun:test";
import { DetectSpdx, NormalizeLicense } from "../Foundry/FoundryBarrel.ts";

// Real-world adversarial fixtures drawn from the exact repos that made GitHub's own matcher return
// NOASSERTION — the ones a naive "permissive words are present" check would wrongly promote. These
// are the permanent regression gate; the legal safety of the whole backfill depends on them.

const MitBody =
  'Permission is hereby granted, free of charge, to any person obtaining a copy\n' +
  'of this software and associated documentation files (the "Software"), to deal\n' +
  'in the Software without restriction, including without limitation the rights\n' +
  'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n' +
  'copies of the Software, and to permit persons to whom the Software is\n' +
  'furnished to do so, subject to the following conditions:\n\n' +
  'The above copyright notice and this permission notice shall be included in all\n' +
  'copies or substantial portions of the Software.\n\n' +
  'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n' +
  'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n' +
  'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n' +
  'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n' +
  'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n' +
  'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n' +
  'SOFTWARE.\n';
const Mit = "MIT License\n\nCopyright (c) 2025 Someone\n\n" + MitBody;

const Bsd3 =
  "Copyright (c) 2010, Ajax.org B.V.\nAll rights reserved.\n\n" +
  "Redistribution and use in source and binary forms, with or without\nmodification, are permitted provided that the following conditions are met:\n" +
  "    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.\n" +
  "    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation.\n" +
  "    * Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.\n\n" +
  "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.\n";

const ApacheNotice =
  "Copyright 2019 Atlassian Pty Ltd\n\n" +
  'Licensed under the Apache License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.\n' +
  "You may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\n" +
  'Unless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\n' +
  "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n";

test("clean permissive licenses are recognized (MIT/BSD-3/Apache-notice)", () => {
  expect(DetectSpdx(Mit)).toMatchObject({ Spdx: "MIT", Permissive: true });
  expect(DetectSpdx(Bsd3)).toMatchObject({ Spdx: "BSD-3-Clause", Permissive: true });
  expect(DetectSpdx(ApacheNotice)).toMatchObject({ Spdx: "Apache-2.0", Permissive: true });
});

test("plain MIT is NOT tripped by its own 'substantial portions of the Software' wording", () => {
  // Regression: an over-broad "portions of" disqualifier once rejected every MIT license.
  expect(NormalizeLicense(Mit)).toContain("substantial portions of the software");
  expect(DetectSpdx(Mit).Permissive).toBe(true);
});

test("MIT + Commons Clause rider is rejected (source-available, NOT permissive)", () => {
  const CommonsClause = Mit + "\n\n\"Commons Clause\" License Condition v1.0\n\nThe Software is provided to you by the Licensor under the License, as defined below, subject to the following condition. Without limiting other conditions in the License, the grant of rights under the License will not include, and the License does not grant to you, the right to Sell the Software.\n";
  expect(DetectSpdx(CommonsClause).Permissive).toBe(false);
});

test("open-core split ('Portions ... are licensed as follows') is rejected", () => {
  const OpenCore = "# License\n\nPortions of this software are licensed as follows:\n\n- All content under packages/ee is licensed under the Enterprise Edition license.\n- All third party components are licensed under their respective licenses.\n- All other content is licensed under the MIT license.\n\n" + MitBody;
  expect(DetectSpdx(OpenCore).Permissive).toBe(false);
});

test("modified Apache (dify-style) is rejected", () => {
  const ModifiedApache = "# Open Source License\n\nDify is licensed under a modified version of the Apache License 2.0, with the following additional conditions:\n\n1. Dify may be used commercially as a backend service for other applications, except for multi-tenant SaaS offerings.\n";
  expect(DetectSpdx(ModifiedApache).Permissive).toBe(false);
});

test("copyleft licenses are rejected (GPL/AGPL/MPL/CC)", () => {
  expect(DetectSpdx("GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n\nThis program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License.").Permissive).toBe(false);
  expect(DetectSpdx("This project is licensed under the AGPL-3.0-or-later License.").Permissive).toBe(false);
  expect(DetectSpdx("Mozilla Public License Version 2.0\n\n1. Definitions").Permissive).toBe(false);
  expect(DetectSpdx("Creative Commons Attribution 4.0 International Public License").Permissive).toBe(false);
});

test("a permissive core buried in a giant bundled-license file is rejected (low coverage)", () => {
  // node.js-style: the repo's own code IS MIT, but the LICENSE file bundles hundreds of dependency
  // licenses. We cannot prove every ingested file is the MIT part, so precision-first rejects it.
  const Filler = "\n\nThis product bundles the following third party software, used under their respective notices.\n".repeat(60);
  const Bundle = "Node.js is licensed for use as follows:\n\n" + MitBody + Filler;
  expect(DetectSpdx(Bundle).Permissive).toBe(false);
});

test("empty / non-license text is rejected", () => {
  expect(DetectSpdx("").Permissive).toBe(false);
  expect(DetectSpdx("See LICENSE file.").Permissive).toBe(false);
  expect(DetectSpdx("GNU-AGPL-3.0.txt").Permissive).toBe(false);
});
