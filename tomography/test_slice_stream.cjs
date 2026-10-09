// Plane controls of the streamed-slices panel: run with `node tomography/test_slice_stream.cjs`.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, 'live.html'), 'utf8');
const context = vm.createContext({});
vm.runInContext(html.slice(html.indexOf('const orientations='), html.indexOf('// Controls and images.')), context);
const plane = (control, shape, size) => JSON.parse(vm.runInContext(
  `JSON.stringify(slicePlane(${JSON.stringify(control)},${JSON.stringify(shape)},${size}))`, context));
const controls = shape => JSON.parse(vm.runInContext(`JSON.stringify(defaultControls(${JSON.stringify(shape)}))`, context));

// streaming.default_planes(rows, columns), which the device uses until a plane is moved.
const defaults = [{"rows": 8, "columns": 64, "size": 64, "planes": [{"origin": [4, 0, 0], "u": [0, 1, 0], "v": [0, 0, 1]}, {"origin": [-28, 32, 0], "u": [1, 0, 0], "v": [0, 0, 1]}, {"origin": [-28, 0, 32], "u": [1, 0, 0], "v": [0, 1, 0]}]}, {"rows": 5, "columns": 33, "size": 33, "planes": [{"origin": [2, 0, 0], "u": [0, 1, 0], "v": [0, 0, 1]}, {"origin": [-14, 16, 0], "u": [1, 0, 0], "v": [0, 0, 1]}, {"origin": [-14, 0, 16], "u": [1, 0, 0], "v": [0, 1, 0]}]}, {"rows": 4, "columns": 11, "size": 11, "planes": [{"origin": [2, 0, 0], "u": [0, 1, 0], "v": [0, 0, 1]}, {"origin": [-3, 5, 0], "u": [1, 0, 0], "v": [0, 0, 1]}, {"origin": [-3, 0, 5], "u": [1, 0, 0], "v": [0, 1, 0]}]}, {"rows": 14, "columns": 6, "size": 14, "planes": [{"origin": [7, -4, -4], "u": [0, 1, 0], "v": [0, 0, 1]}, {"origin": [0, 3, -4], "u": [1, 0, 0], "v": [0, 0, 1]}, {"origin": [0, -4, 3], "u": [1, 0, 0], "v": [0, 1, 0]}]}];
for (const {rows, columns, size, planes} of defaults) {
  const shape = [rows, columns, columns];
  assert.deepEqual(controls(shape).map(control => plane(control, shape, size)), planes);
}

const close = (actual, expected) => actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 1e-9, `${actual} != ${expected}`));
const shape = [8, 64, 64], size = 64, centre = [3.5, 31.5, 31.5];
const middle = ({origin, u, v}) => origin.map((value, index) => value + (size - 1) / 2 * (u[index] + v[index]));
for (const control of [{orientation: 'axial', position: 2, tilt: 30, turn: -20, zoom: 1},
                       {orientation: 'coronal', position: 40, tilt: -75, turn: 10, zoom: 4},
                       {orientation: 'sagittal', position: 9, tilt: 0, turn: 90, zoom: .5}]) {
  const result = plane(control, shape, size), axis = ['axial', 'coronal', 'sagittal'].indexOf(control.orientation);
  // Tilting and zooming keep the image centre on the chosen position, and the steps orthogonal.
  const expected = [...centre]; expected[axis] = control.position;
  close(middle(result), expected);
  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  close([dot(result.u, result.v), dot(result.u, result.u), dot(result.v, result.v)], [0, 1 / control.zoom ** 2, 1 / control.zoom ** 2]);
}
// A quarter turn of the axial plane about its horizontal axis looks along y: it becomes coronal.
close(plane({orientation: 'axial', position: 4, tilt: 90, turn: 0, zoom: 1}, shape, size).u, [1, 0, 0]);
console.log('Slice plane controls match the device defaults and keep their pivot.');
