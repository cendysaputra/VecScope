// Usage: node scripts/screenshots.js [url] [outdir]
const { chromium } = require('playwright');

const url = process.argv[2] || 'http://localhost:8501';
const out = process.argv[3] || 'docs/screenshots';

const TABS = [
  ['Tokens', 'tokens'],
  ['Vector strips', 'strips'],
  ['Similarity', 'similarity'],
  ['2D map', 'map2d'],
  ['Analogy', 'analogy'],
  ['Images (CLIP)', 'clip'],
];

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('pageerror', e.message));
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.getByRole('tab', { name: /Tokens/ }).waitFor({ timeout: 180000 });
  for (const [label, file] of TABS) {
    await page.setViewportSize({ width: 1400, height: 1000 });
    await page.getByRole('tab', { name: new RegExp(label.replace(/[()]/g, '\\$&')) }).click();
    if (file === 'clip') {
      await page.getByText('Enable CLIP').click();
      await page.getByText('Mixed 2D map').waitFor({ timeout: 600000 });
      if (process.env.CLOSE_GAP) await page.getByText('Close the modality gap').click();
    }
    await page.waitForTimeout(2500);
    // Wait for rerun
    await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 600000 }).catch(() => {});
    const panel = page.locator('[role="tabpanel"]:visible').first();
    // Fit whole panel
    const h = await panel.evaluate((el) => el.scrollHeight);
    await page.setViewportSize({ width: 1400, height: h + 400 });
    await page.waitForTimeout(2500);
    const errors = await panel.locator('[data-testid="stException"]').count();
    if (errors) console.error(`!! ${label}: ${errors} exception(s)`);
    await panel.screenshot({ path: `${out}/${file}.png` });
    console.log(`saved ${out}/${file}.png`);
  }
  await browser.close();
})();
