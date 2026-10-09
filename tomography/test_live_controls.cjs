const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, 'live.html'), 'utf8');
const elements = new Map();
function element() {
  return {
    value: '', dataset: {}, classList: {toggle() {}},
    append() {}, insertBefore() {}, addEventListener() {}, removeAttribute() {},
  };
}
const context = vm.createContext({
  document: {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, element());
      return elements.get(selector);
    },
    querySelectorAll() { return []; },
    createElement: element,
    createTextNode(text) { return text; },
  },
  setTimeout() { context.polls++; },
  polls: 0,
  fetch: async () => ({ok: true, json: async () => context.status}),
});
vm.runInContext(html.slice(html.indexOf("'use strict';"), html.indexOf('// Both renderers')), context);
vm.runInContext(`
  let frames=[], selected=null, displayed=null, followLive=true, cache=new Map(), cacheBytes=0;
  let controller=null, requestID=0, busy=false, sliceController=null, sliceRequestID=0;
  let sliceURL=null, sliceShape=null, inputDefaultsSet=false, phase='running';
  function stopPlayback(){}
  function refreshControls(){}
  function updateDisplayRange(){}
  let inspectionData=null, inspectionShape=null, inspectionPoint=[0,0,0];
  function drawInspection(){}
`, context);
vm.runInContext(html.slice(html.indexOf('function resetViewerRun('), html.lastIndexOf('\nupdate();')), context);

function pipeline(runID, revision, pixels, phase = 'running', active = null) {
  const requested = {revision, options: {pixels, slices: 8, angles: 96}};
  return {run_id: runID, phase, requested, active: active || requested};
}
function reconstruction(algorithm, revision) {
  const requested = {revision, options: {algorithm}};
  return {requested, active: {...requested, scan_id: 1}, detector_columns: 128};
}
async function poll(pipelineState, reconstructionState) {
  context.status = {pipeline_control: pipelineState, reconstruction_control: reconstructionState, viewer: {}};
  await vm.runInContext('update()', context);
  assert.notEqual(elements.get('#phase').textContent, 'Disconnected');
}

(async () => {
  await poll(pipeline(0, 0, 64), reconstruction('gridrec', 3));
  const queued = pipeline(1, 1, 128, 'restarting', pipeline(0, 0, 64).active);
  await poll(queued, reconstruction('gridrec', 3));
  assert.equal(elements.get('#pipeline-pixels').value, 128);
  assert.equal(vm.runInContext('viewerRunID', context), 0);
  assert.equal(elements.get('#recon-fields').disabled, true);

  await poll(pipeline(1, 1, 128), reconstruction('fbp', 0));
  assert.equal(elements.get('#pipeline-pixels').value, 128);
  assert.equal(elements.get('#recon-algorithm').value, 'fbp');
  assert.equal(vm.runInContext('viewerRunID', context), 1);

  await poll(pipeline(0, 0, 64), reconstruction('gridrec', 3));
  assert.equal(elements.get('#pipeline-pixels').value, 128);
  assert.equal(elements.get('#recon-algorithm').value, 'fbp');
  assert.equal(vm.runInContext('viewerRunID', context), 1);
  assert.equal(context.polls, 4);

  await poll(pipeline(1, 1, 128), reconstruction('sirt', 1));
  assert.equal(elements.get('#pipeline-pixels').value, 128);
  assert.equal(elements.get('#recon-algorithm').value, 'sirt');

  elements.get('#pipeline-pixels').value = '256';
  vm.runInContext('pipelineDirty=true', context);
  await poll(pipeline(1, 1, 128), reconstruction('fbp', 2));
  assert.equal(elements.get('#pipeline-pixels').value, '256');
  assert.equal(elements.get('#recon-algorithm').value, 'fbp');
  vm.runInContext('pipelineDirty=false', context);
  const disabled = pipeline(2, 2, 128);
  disabled.active.options.saving = false;
  disabled.active.options.decompression = false;
  await poll(disabled, reconstruction('fbp', 0));
  assert.equal(elements.get('#pipeline-saving').checked, false);
  assert.equal(elements.get('#pipeline-decompression').checked, false);
  assert.equal(vm.runInContext('pipelineFormOptions().saving', context), false);
  assert.equal(vm.runInContext('pipelineFormOptions().decompression', context), false);
  vm.runInContext('fillPipelineOptions({saving:true,decompression:true})', context);
  assert.equal(vm.runInContext('pipelineFormOptions().saving', context), true);
  assert.equal(vm.runInContext('pipelineFormOptions().decompression', context), true);
  console.log('Live controls retain geometry and algorithms across restarts and stale polls.');
})().catch(error => { console.error(error); process.exitCode = 1; });
