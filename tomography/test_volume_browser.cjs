/* Optional browser regression: known inner objects must survive an opaque shell.
 * Set TOMOGRAPHY_CHROME to a Chrome executable and install puppeteer-core,
 * or set TOMOGRAPHY_PUPPETEER to its module path. Run with webgpu or fallback.
 */
const assert = require('node:assert/strict');
const {spawn} = require('node:child_process');
const path = require('node:path');
const puppeteer = require(process.env.TOMOGRAPHY_PUPPETEER || 'puppeteer-core');
const mode = process.argv[2] || 'webgpu';
const python = process.env.TOMOGRAPHY_TEST_PYTHON || path.resolve(__dirname, '../.pixi/envs/default/bin/python');
const fixture = `
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import numpy as np
from reconstruction import configuration
from viewer import ReconstructionControl, VolumeHistory, make_handler
class Proxy:
    def __init__(self):
        self.state = dict(requested=dict(options=configuration('gridrec'), revision=0),
                          active=dict(options=configuration('gridrec'), revision=0, scan_id=1),
                          detector_columns=64, finished=False)
    def command_inout(self, command, argument=None):
        if command == 'ConfigureReconstruction':
            options = json.loads(argument)
            if options != self.state['requested']['options']:
                self.state['requested'] = dict(options=options, revision=self.state['requested']['revision'] + 1)
        return json.dumps(self.state)
proxy = Proxy()
history = VolumeHistory()
history.append(np.arange(32, dtype=np.float32).reshape(2, 4, 4) * .0001, 1, 0, dict(skipped=0, transport='fixture'), proxy.state['active'])
Base = make_handler(history, Path('/tmp'), ReconstructionControl(proxy))
class Handler(Base):
    def do_GET(self):
        if self.path == '/api/status':
            self.respond(json.dumps(dict(phase='running', viewer=history.snapshot(), reconstruction_control=proxy.state)).encode(), 'application/json')
        else:
            super().do_GET()
    def do_POST(self):
        if self.path == '/api/advance':
            scan_id = proxy.state['active']['scan_id'] + 1
            proxy.state['active'] = dict(proxy.state['requested'], scan_id=scan_id)
            history.append(np.zeros((2, 4, 4), dtype=np.float32), scan_id, scan_id - 1,
                           dict(skipped=0, transport='fixture'), proxy.state['active'])
            self.respond(b'{}', 'application/json')
        elif self.path == '/api/finish':
            proxy.state['finished'] = True
            self.respond(b'{}', 'application/json')
        else:
            super().do_POST()
server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
`;
const server = spawn(python, ['-u', '-c', fixture], {cwd: __dirname});
server.stderr.on('data', data => process.stderr.write(data));
(async () => {
  let browser;
  try {
    const port = await new Promise((resolve, reject) => {
      server.stdout.once('data', data => resolve(Number(data.toString().trim())));
      server.once('error', reject);
      server.once('exit', code => reject(new Error(`Fixture exited: ${code}`)));
    });
    browser = await puppeteer.launch({
      executablePath: process.env.TOMOGRAPHY_CHROME, headless: true,
      args: ['--no-sandbox', '--use-gl=angle', '--use-angle=vulkan', '--enable-unsafe-swiftshader',
        '--enable-unsafe-webgpu', '--use-vulkan=swiftshader', '--enable-features=Vulkan', '--disable-dev-shm-usage'],
    });
    const page = await browser.newPage(), errors = [];
    page.on('pageerror', error => errors.push(error.message));
    if (mode === 'fallback') await page.evaluateOnNewDocument(() => Object.defineProperty(navigator, 'gpu', {value: undefined}));
    await page.setViewport({width: 1000, height: 900});
    await page.goto(`http://127.0.0.1:${port}`);
    await page.waitForFunction(() => displayed?.scan_id === 1);
    assert.equal(await page.evaluate(() => renderer.backend), mode === 'fallback' ? 'WebGL 2' : 'WebGPU');
    assert.equal(await page.evaluate(() => inspectionData instanceof Uint16Array), true);
    await page.select('#view-mode', 'inspection');
    await page.waitForFunction(() => slicePanels[0].canvas.width > 1);
    const samples = await page.evaluate(() => {
      document.querySelector('#window-high').value = '.01';
      document.querySelector('#show-crosshair').checked = false;
      drawInspection();
      return slicePanels.map(panel => {
        const [width, height] = panel.imageSize;
        const scale = Math.min(panel.canvas.width / width, panel.canvas.height / height);
        const left = (panel.canvas.width - width * scale) / 2;
        const top = (panel.canvas.height - height * scale) / 2;
        return panel.canvas.getContext('2d').getImageData(Math.floor(left + .5 * scale), Math.floor(top + .5 * scale), 1, 1).data[0];
      });
    });
    assert.deepEqual(samples, [41, 20, 5]);
    await page.evaluate(() => {document.querySelector('#show-crosshair').checked = true; drawInspection();});
    await page.evaluate(() => {
      slicePanels[0].slider.value = '0';
      slicePanels[0].slider.dispatchEvent(new Event('input'));
    });
    assert.deepEqual(await page.evaluate(() => inspectionPoint), [0, 2, 2]);
    const axial = await page.$('#inspection canvas');
    await axial.evaluate(element => element.scrollIntoView({block: 'center'}));
    const axialBox = await axial.boundingBox();
    const axialImageWidth = Math.min(axialBox.width, axialBox.height);
    await page.mouse.click(axialBox.x + (axialBox.width - axialImageWidth) / 2 + axialImageWidth * .375, axialBox.y + axialBox.height * .375);
    assert.deepEqual(await page.evaluate(() => inspectionPoint), [0, 1, 1]);
    await page.evaluate(() => {
      document.querySelector('#spacing-z').value = '5';
      document.querySelector('#spacing-unit').value = 'µm';
      document.querySelector('#spacing-unit').dispatchEvent(new Event('change'));
    });
    assert.deepEqual(await page.evaluate(() => renderer.extent), [.2, .2, .5]);
    assert.match(await page.$eval('#point-position', element => element.textContent), /µm/);
    await page.click('#focus-point');
    assert.deepEqual(await page.evaluate(() => renderer.target), await page.evaluate(() => renderer.point()));
    await page.click('#clip-point');
    assert.equal(await page.$eval('#clip-enabled', element => element.checked), true);
    assert.equal(await page.$eval('#cut', element => Number(element.value)), .25);
    const volumeCanvas = await page.$('#volume');
    await volumeCanvas.evaluate(element => element.scrollIntoView({block: 'center'}));
    const volumeBox = await volumeCanvas.boundingBox();
    const targetBefore = await page.evaluate(() => [...renderer.target]);
    await page.keyboard.down('Shift');
    await page.mouse.move(volumeBox.x + volumeBox.width / 2, volumeBox.y + volumeBox.height / 2);
    await page.mouse.down();
    await page.mouse.move(volumeBox.x + volumeBox.width / 2 + 40, volumeBox.y + volumeBox.height / 2 + 20);
    await page.mouse.up(); await page.keyboard.up('Shift');
    assert.notDeepEqual(await page.evaluate(() => renderer.target), targetBefore);
    await page.evaluate(() => renderer.zoom(-10));
    assert.ok(await page.evaluate(() => renderer.distance < .5));
    for (const axis of ['0', '1', '2']) {
      await page.select('#clip-axis', axis);
      for (const side of ['0', '1']) await page.select('#clip-side', side);
    }
    await page.select('#spacing-unit', 'voxel');
    await page.click('#reset');
    await page.select('#view-mode', 'volume');
    await page.evaluate(() => {
      renderer.device?.addEventListener('uncapturederror', event => window.renderError = event.error.message);
      window.testVolume = async (size, features) => {
        const shape = [size / 2, size, size], [z, y, x] = shape;
        const voxels = new Uint16Array(z * y * x);
        for (let iz = 0; iz < z; iz++) for (let iy = 0; iy < y; iy++) for (let ix = 0; ix < x; ix++) {
          const px = (ix + .5) / x * 2 - 1, py = (iy + .5) / y * 2 - 1, pz = (iz + .5) / z * 2 - 1;
          const radius = Math.hypot(px, py, pz);
          // Exact constant-intensity regions: interpolation must not invent features.
          let value = radius < .85 ? 51 : radius < .96 ? 255 : 0;
          if (features && radius < .85) {
            if (Math.hypot(px + .27, py - .15, pz + .1) < .17) value = 77;
            if (Math.hypot(px - .27, py + .15, pz - .1) < .17) value = 102;
          }
          voxels[(iz * y + iy) * x + ix] = value * 257;
        }
        await renderer.setVolume(voxels, shape);
        if (renderer.device) await renderer.device.queue.onSubmittedWorkDone();
      };
    });
    async function warmPixels() {
      const png = await (await page.$('#volume')).screenshot();
      return page.evaluate(async encoded => {
        const image = new Image(); image.src = 'data:image/png;base64,' + encoded; await image.decode();
        const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
        const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let count = 0;
        for (let i = 0; i < pixels.length; i += 4) if (pixels[i] > 70 && pixels[i] > pixels[i + 2] * 1.5) count++;
        return count;
      }, Buffer.from(png).toString('base64'));
    }
    const counts = [];
    for (const size of [64, 128]) {
      await page.evaluate(size => window.testVolume(size, false), size);
      const empty = await warmPixels();
      assert.ok(empty < 30, `Shell must not invent warm interior features: ${empty}`);
      await page.evaluate(size => window.testVolume(size, true), size);
      const features = await warmPixels();
      assert.ok(features > 300, `Known interior features must remain visible: ${features}`);
      counts.push({size, empty, features});
    }
    await page.select('#appearance', '0');
    await page.focus('#volume'); await page.keyboard.press('ArrowRight');
    assert.equal(await page.evaluate(() => renderer.gl?.getError() || 0), 0);
    await page.click('#reset');
    assert.equal(await page.$eval('#appearance', element => element.value), '1');
    assert.ok(await warmPixels() > 300);
    if (mode === 'webgpu') {
      assert.equal(await page.evaluate(() => window.renderError), undefined);
      await page.evaluate(() => renderer.device.destroy());
      await page.waitForFunction(() => renderer?.backend === 'WebGL 2' && displayed);
      assert.equal(await page.$eval('#appearance', element => element.value), '1');
      await page.evaluate(() => window.testVolume(64, true));
      assert.ok(await warmPixels() > 300);
    }
    await page.waitForFunction(() => !document.querySelector('#recon-fields').disabled);
    assert.equal(await page.$eval('#recon-algorithm', element => element.value), 'gridrec');
    await page.select('#recon-algorithm', 'sirt');
    await page.evaluate(() => {
      document.querySelector('#recon-iterations').value = '12';
      document.querySelector('#recon-min_constraint').value = '';
      document.querySelector('#recon-iterations').dispatchEvent(new Event('input', {bubbles: true}));
    });
    // Repeated live polls must preserve unapplied edits.
    await new Promise(resolve => setTimeout(resolve, 700));
    assert.equal(await page.$eval('#recon-algorithm', element => element.value), 'sirt');
    assert.equal(await page.$eval('#recon-iterations', element => element.value), '12');
    await page.click('#recon-apply');
    await page.waitForFunction(() => reconState?.requested.revision === 1 && !reconSaving);
    assert.equal(await page.evaluate(() => reconState.active.options.algorithm), 'gridrec');
    assert.equal(await page.evaluate(() => reconState.requested.options.iterations), 12);
    assert.equal(await page.evaluate(() => reconState.requested.options.min_constraint), null);
    assert.equal(await page.evaluate(() => reconState.requested.options.filter), null);
    assert.match(await page.$eval('#recon-status', element => element.textContent), /queued/);
    await page.evaluate(() => fetch('/api/advance', {method: 'POST'}));
    await page.waitForFunction(() => reconState?.active.revision === 1 && displayed?.scan_id === 2);
    assert.match(await page.$eval('#displayed-reconstruction', element => element.textContent), /SIRT.*settings 1/);
    await page.select('#recon-algorithm', 'fbp');
    await page.select('#recon-filter', 'hann');
    assert.equal(await page.$eval('#recon-cutoff-control', element => element.hidden), false);
    await page.evaluate(() => {document.querySelector('#recon-filter_cutoff').value = '.7';});
    await page.click('#recon-apply');
    await page.waitForFunction(() => reconState?.requested.revision === 2 && !reconSaving);
    assert.equal(await page.evaluate(() => reconState.requested.options.filter_cutoff), .7);
    await page.select('#recon-filter', 'ram-lak');
    assert.equal(await page.$eval('#recon-cutoff-control', element => element.hidden), true);
    await page.click('#recon-apply');
    await page.waitForFunction(() => reconState?.requested.revision === 3 && !reconSaving);
    assert.equal(await page.evaluate(() => reconState.requested.options.filter_cutoff), null);
    assert.equal(await page.evaluate(() => reconState.active.revision), 1);
    await page.evaluate(() => fetch('/api/finish', {method: 'POST'}));
    await page.waitForFunction(() => document.querySelector('#recon-fields').disabled);
    assert.equal(await page.$eval('#recon-apply', element => element.disabled), true);
    assert.match(await page.$eval('#recon-status', element => element.textContent), /queued settings were not used/);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({mode, counts, controls: true, reconstructionControls: true, pageErrors: errors}));
  } finally { await browser?.close(); server.kill(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
