# ORBIT-Q Task Atlas

Interactive companion to the offline ORBIT-Q task-survey report. The site and
offline report both read the reviewed `../survey.json` snapshot.

Requires Node.js 22.13 or newer.

```bash
npm install
npm test -- --run
npm run build
node --test tests/rendered-html.test.mjs
npm run dev
```

The app is built with the Sites vinext starter and declares no D1 or R2
bindings. Deployment metadata lives in `.openai/hosting.json`.
