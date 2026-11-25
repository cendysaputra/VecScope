// Usage: node scripts/demo_gif.js [url] [out.gif]
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const url = process.argv[2] || 'http://localhost:8501';
const out = process.argv[3] || 'docs/demo.gif';
const W = 1280;
const H = 800;

// Fake cursor
const CURSOR = () => {
  const add = () => {
    const c = document.createElement('div');
    c.style.cssText = 'position:fixed;left:0;top:0;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;'
      + 'background:rgba(11,11,11,.35);border:2px solid #fff;box-shadow:0 0 0 1px rgba(11,11,11,.5);'
      + 'pointer-events:none;z-index:2147483647;transition:transform .12s;';
    document.body.appendChild(c);
    addEventListener('mousemove', (e) => { c.style.left = `${e.clientX}px`; c.style.top = `${e.clientY}px`; }, true);
    addEventListener('mousedown', () => { c.style.transform = 'scale(.7)'; }, true);
    addEventListener('mouseup', () => { c.style.transform = ''; }, true);
  };
  if (document.body) add(); else addEventListener('DOMContentLoaded', add);
};

async function settle(page, ms = 700) {
  await page.waitForTimeout(ms);
  await page.waitForFunction(() => !document.querySelector('[data-testid="stStatusWidget"]'), null, { timeout: 300000 }).catch(() => {});
  await page.waitForTimeout(300);
}

async function moveTo(page, locator) {
  await locator.scrollIntoViewIfNeeded();
  const b = await locator.boundingBox();
  await page.mouse.move(b.x + Math.min(b.width / 2, 120), b.y + b.height / 2, { steps: 18 });
  await page.waitForTimeout(150);
}

async function click(page, locator) {
  await moveTo(page, locator);
  await page.mouse.down();
  await page.mouse.up();
}

async function typeInto(page, locator, text, submit) {
  await click(page, locator);
  await page.keyboard.press('Control+A');
  await page.keyboard.type(text, { delay: 55 });
  await page.keyboard.press(submit);
  await settle(page);
}

async function scroll(page, dy) {
  await page.mouse.move(W * 0.62, H * 0.6, { steps: 8 });
  for (let i = 0; i < 10; i++) {
    await page.mouse.wheel(0, dy / 10);
    await page.waitForTimeout(40);
  }
  await page.waitForTimeout(400);
}

async function tab(page, name) {
  await scroll(page, -3000);
  await click(page, page.getByRole('tab', { name }));
  await settle(page, 700);
  await scroll(page, 260);
}

(async () => {
  const browser = await chromium.launch();

  // Warm caches
  const warm = await browser.newPage({ viewport: { width: W, height: H } });
  await warm.goto(url, { waitUntil: 'networkidle' });
  await warm.getByText('19 tokens').waitFor({ timeout: 300000 });
  await warm.getByRole('tab', { name: /Images/ }).click();
  await warm.getByText('Enable CLIP').click();
  await warm.getByText('Text and images on one map').waitFor({ timeout: 600000 });
  await warm.close();

  // Record run
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vs-demo-'));
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir, size: { width: W, height: H } } });
  await ctx.addInitScript(CURSOR);
  const t0 = Date.now();
  const page = await ctx.newPage();
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.getByText('19 tokens').waitFor({ timeout: 300000 });
  await settle(page, 800);
  const start = (Date.now() - t0) / 1000;
  await page.waitForTimeout(1500);

  // Tokens
  await typeInto(page, page.getByLabel('Text', { exact: true }), 'Embeddings turn unbelievable words into numbers.', 'Control+Enter');
  await scroll(page, 380);
  await page.waitForTimeout(2600);

  // Vector strips
  await tab(page, /Vector strips/);
  await typeInto(page, page.getByLabel('Words (comma or space separated)'), 'king, man, woman, queen, prince', 'Enter');
  await page.waitForTimeout(2400);

  // 2D map
  await tab(page, /2D map/);
  await scroll(page, 200);
  await page.waitForTimeout(2600);

  // Analogy
  await tab(page, /Analogy/);
  await click(page, page.getByRole('tabpanel').getByTestId('stSelectbox').first());
  await page.waitForTimeout(400);
  await click(page, page.getByRole('option', { name: 'paris - france + italy' }));
  await settle(page);
  await page.waitForTimeout(2600);

  // CLIP search
  await tab(page, /Images/);
  await click(page, page.getByText('Enable CLIP'));
  await page.getByText('Search images with text').waitFor({ timeout: 300000 });
  await settle(page);
  await typeInto(page, page.getByLabel('Query'), 'a bee on a flower', 'Enter');
  await scroll(page, 300);
  await page.waitForTimeout(3000);

  const video = await page.video().path();
  await ctx.close();
  await browser.close();

  // Encode GIF
  const vf = 'fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];'
    + '[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle';
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-ss', String(start), '-i', video, '-vf', vf, '-loop', '0', out]);
  fs.rmSync(dir, { recursive: true, force: true });
  console.log(`saved ${out} (${(fs.statSync(out).size / 1e6).toFixed(1)} MB)`);
})();
