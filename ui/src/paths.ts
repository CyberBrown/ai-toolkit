import path from 'path';
export const TOOLKIT_ROOT = process.env.TOOLKIT_ROOT || path.resolve(__dirname, '..', '..');
export const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
export const defaultDatasetsFolder = path.join(TOOLKIT_ROOT, 'datasets');
export const defaultDataRoot = path.join(TOOLKIT_ROOT, 'data');
