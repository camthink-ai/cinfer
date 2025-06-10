export { isRequestSuccess, getResponseData, awaitWrap, pLimit, API_PREFIX } from './client';

export { default as globalAPI, type GlobalAPISchema } from './global';
export {
    default as dashboardAPI,
    type DashboardAPISchema,
    type SystemMetricsType,
} from './dashboard';
